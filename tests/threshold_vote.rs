//! A vote cast by a key held as FROST shares.
//!
//! The vote proof witnesses only the voting key's `ak`, so a 2-of-3 FROST
//! group can vote: anyone holding the full viewing key builds the proof, and
//! two participants sign the vote under the proof's `r_vpk`. Neither proving
//! nor signing uses the spending scalar. (The test's trusted dealer knows it
//! for a moment; a DKG yields the same key and public key packages.)
//!
//! Signatures are checked with both RedPallas verifiers vote-sdk can be built
//! with (`circuits/src/redpallas.rs`): `zakura-reddsa` and `reddsa` 0.6. The signed message stands in
//! for the cast-vote sighash, which is computed outside this crate.

use std::collections::BTreeMap;

use frost_rerandomized::frost_core::{Ciphersuite, Field, Group};
use frost_rerandomized::{RandomizedParams, Randomizer};
use rand_core_06::OsRng;
use reddsa_frost::frost::redpallas::{self, keys::EvenY, PallasBlake2b512};
use voting_circuits::{
    ff::PrimeField,
    group::{Curve, GroupEncoding},
    spend_auth_g_affine,
    vote_proof::{
        build_vote_proof_from_fvk, derive_vote_authority_transition_from_fvk, verify_vote_proof,
        VotePrfKey, VoteProofBundle,
    },
    MAX_PROPOSAL_AUTHORITY, VOTE_COMM_TREE_DEPTH,
};
use voting_crypto_deps::orchard::keys::{FullViewingKey, SpendingKey};
use voting_crypto_deps::pasta_curves::pallas;

type FrostScalarField = <<PallasBlake2b512 as Ciphersuite>::Group as Group>::Field;

struct ThresholdKey {
    key_packages: BTreeMap<redpallas::Identifier, redpallas::keys::KeyPackage>,
    public_key_package: redpallas::keys::PublicKeyPackage,
    fvk: FullViewingKey,
}

fn threshold_key() -> ThresholdKey {
    let (shares, public_key_package) = redpallas::keys::generate_with_dealer(
        3,
        2,
        redpallas::keys::IdentifierList::Default,
        OsRng,
    )
    .expect("dealer keygen");
    assert!(
        public_key_package.has_even_y(),
        "the RedPallas ciphersuite normalises the group key, as Orchard's ak requires"
    );
    let key_packages = shares
        .into_iter()
        .map(|(id, share)| {
            (
                id,
                redpallas::keys::KeyPackage::try_from(share).expect("key package"),
            )
        })
        .collect();

    let ak: [u8; 32] = public_key_package
        .verifying_key()
        .serialize()
        .expect("group key")
        .try_into()
        .expect("32-byte ak");
    let template = FullViewingKey::from(&SpendingKey::from_bytes([0x42; 32]).unwrap()).to_bytes();
    let mut fvk_bytes = [0u8; 96];
    fvk_bytes[..32].copy_from_slice(&ak);
    fvk_bytes[32..].copy_from_slice(&template[32..]);
    let fvk = FullViewingKey::from_bytes(&fvk_bytes).expect("FVK with the group's ak");

    ThresholdKey {
        key_packages,
        public_key_package,
        fvk,
    }
}

fn random_alpha() -> [u8; 32] {
    FrostScalarField::serialize(&FrostScalarField::random(&mut OsRng))
}

fn alpha_v(alpha: [u8; 32]) -> pallas::Scalar {
    Option::from(pallas::Scalar::from_repr(alpha)).expect("same scalar field")
}

fn threshold_sign(
    key: &ThresholdKey,
    signers: &[redpallas::Identifier],
    alpha: [u8; 32],
    message: &[u8],
) -> [u8; 64] {
    let alpha = FrostScalarField::deserialize(&alpha).expect("canonical alpha");
    let mut nonces = BTreeMap::new();
    let mut commitments = BTreeMap::new();
    for id in signers {
        let (n, c) = redpallas::round1::commit(key.key_packages[id].signing_share(), &mut OsRng);
        nonces.insert(*id, n);
        commitments.insert(*id, c);
    }
    let signing_package = redpallas::SigningPackage::new(commitments, message);

    let mut shares = BTreeMap::new();
    for id in signers {
        #[allow(deprecated)]
        let share = frost_rerandomized::sign::<PallasBlake2b512>(
            &signing_package,
            &nonces[id],
            &key.key_packages[id],
            Randomizer::from_scalar(alpha),
        )
        .expect("signature share");
        shares.insert(*id, share);
    }
    let params = RandomizedParams::from_randomizer(
        key.public_key_package.verifying_key(),
        Randomizer::from_scalar(alpha),
    );
    frost_rerandomized::aggregate(&signing_package, &shares, &key.public_key_package, &params)
        .expect("aggregate")
        .serialize()
        .expect("signature")
        .try_into()
        .expect("64-byte signature")
}

fn verifies_like_vote_chain(r_vpk: [u8; 32], message: &[u8], signature: [u8; 64]) -> bool {
    let zakura =
        reddsa_zakura::VerificationKey::<reddsa_zakura::orchard::SpendAuth>::try_from(r_vpk)
            .expect("r_vpk is a valid key")
            .verify(message, &reddsa_zakura::Signature::from(signature))
            .is_ok();
    let lrz = reddsa_lrz::VerificationKey::<reddsa_lrz::orchard::SpendAuth>::try_from(r_vpk)
        .expect("r_vpk is a valid key")
        .verify(message, &reddsa_lrz::Signature::from(signature))
        .is_ok();
    assert_eq!(zakura, lrz, "both vote-chain verifier backends agree");
    zakura
}

fn prove_vote(key: &ThresholdKey, alpha: [u8; 32]) -> VoteProofBundle {
    build_vote_proof_from_fvk(
        &key.fvk,
        &VotePrfKey::from_bytes([7; 32]),
        1,
        12_500_000,
        pallas::Base::from(0xDEAD_u64),
        pallas::Base::from(0xCAFE_u64),
        [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH],
        0,
        123,
        50,
        1,
        (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine(),
        alpha_v(alpha),
        MAX_PROPOSAL_AUTHORITY,
        true,
    )
    .expect("vote proof from the group's FVK")
}

#[test]
#[ignore = "expensive end-to-end proof generation; run with --ignored --release"]
fn frost_group_casts_a_vote_without_a_spending_key() {
    let key = threshold_key();
    let alpha = random_alpha();
    let bundle = prove_vote(&key, alpha);

    verify_vote_proof(&bundle.proof, &bundle.instance).expect("vote proof verifies");

    let transition = derive_vote_authority_transition_from_fvk(
        &key.fvk,
        1,
        12_500_000,
        pallas::Base::from(0xDEAD_u64),
        pallas::Base::from(0xCAFE_u64),
        50,
        MAX_PROPOSAL_AUTHORITY,
    )
    .expect("transition");
    assert_eq!(
        bundle.instance.vote_authority_note_new,
        transition.vote_authority_note_new
    );

    let cast_vote_digest = [0x5a; 32];
    let ids: Vec<_> = key.key_packages.keys().copied().collect();
    for pair in [[ids[0], ids[1]], [ids[0], ids[2]], [ids[1], ids[2]]] {
        let signature = threshold_sign(&key, &pair, alpha, &cast_vote_digest);
        assert!(
            verifies_like_vote_chain(bundle.r_vpk_bytes, &cast_vote_digest, signature),
            "any two of three sign the vote under the proof's r_vpk"
        );
    }

    let wrong_alpha = threshold_sign(&key, &ids[..2], random_alpha(), &cast_vote_digest);
    assert!(!verifies_like_vote_chain(
        bundle.r_vpk_bytes,
        &cast_vote_digest,
        wrong_alpha
    ));
}

#[test]
fn group_r_vpk_matches_the_frost_randomized_key() {
    let key = threshold_key();
    let alpha = random_alpha();
    let ak: [u8; 32] = key.fvk.to_bytes()[..32].try_into().unwrap();
    let ak = Option::<pallas::Point>::from(pallas::Point::from_bytes(&ak)).expect("ak point");
    let r_vpk = (ak + spend_auth_g_affine() * alpha_v(alpha)).to_affine();

    let params = RandomizedParams::<PallasBlake2b512>::from_randomizer(
        key.public_key_package.verifying_key(),
        Randomizer::from_scalar(FrostScalarField::deserialize(&alpha).unwrap()),
    );
    let frost_r_vpk: [u8; 32] = params
        .randomized_verifying_key()
        .serialize()
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(r_vpk.to_bytes(), frost_r_vpk);

    let digest = [0x11; 32];
    let ids: Vec<_> = key.key_packages.keys().copied().collect();
    let signature = threshold_sign(&key, &ids[..2], alpha, &digest);
    assert!(verifies_like_vote_chain(frost_r_vpk, &digest, signature));
}
