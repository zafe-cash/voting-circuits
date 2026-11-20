//! Address ownership gadget: shared CommitIvk + pk_d derivation.
//!
//! Encapsulates the common "address ownership" and optional "SpendAuthG mul"
//! logic used by:
//!
//! - **ZKP #1 (delegation)**: Condition 4 (spend authority: `[alpha]*SpendAuthG + ak_P` → rk),
//!   Condition 5 (CommitIvk & diversified address: `pk_d_signed = [ivk]*g_d_signed`).
//! - **ZKP #2 (vote proof)**: Condition 3 (spend authority: `ak = ExtractP(vsk_ak)`, a witnessed point;
//!   address ownership: `vpk_pk_d = [ivk_v]*vpk_g_d` via CommitIvk).
//!
//! The same CommitIvk + pk_d derivation pattern is used in the main Orchard action
//! circuit for spend authority and diversified address integrity.

use voting_crypto_deps::composition::{
    CommitIvkChip, OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains,
};
use voting_crypto_deps::halo2_gadgets::{
    ecc::{chip::EccChip, NonIdentityPoint, ScalarFixed, ScalarVar},
    sinsemilla::chip::SinsemillaChip,
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::Error,
};
use voting_crypto_deps::pasta_curves::pallas;

// ================================================================
// Prove address ownership (CommitIvk + [ivk]*g_d → constrain pk_d)
// ================================================================

/// Proves address ownership: `ivk = CommitIvk(ak, nk, rivk)`, then
/// `pk_d_claimed = [ivk] * g_d` (constrain derived to claimed).
///
/// Same constraint flow as the Orchard action circuit: CommitIvk yields ivk,
/// then variable-base mul and equality constraint. Used by ZKP #1 (delegation,
/// condition 5) and ZKP #2 (vote proof, condition 3).
///
/// # Arguments
///
/// * `sinsemilla_chip` – Sinsemilla chip used by CommitIvk.
/// * `ecc_chip` – ECC chip for scalar mul and equality.
/// * `commit_ivk_chip` – CommitIvk chip (canonicity gates).
/// * `layouter` – Circuit layouter.
/// * `label` – Namespace label for the region.
/// * `ak` – Spend validating key x-coordinate (`ExtractP` of a witnessed `ak_P` / `vsk_ak`).
/// * `nk` – Nullifier deriving key.
/// * `rivk` – CommitIvk randomness.
/// * `g_d` – Diversified base (non-identity point).
/// * `pk_d_claimed` – Claimed diversified transmission key; constrained to equal `[ivk]*g_d`.
///
/// Returns the ivk cell so callers (e.g. delegation) can reuse it for per-note
/// diversified address checks.
pub(crate) fn prove_address_ownership(
    sinsemilla_chip: SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    ecc_chip: EccChip<OrchardFixedBases>,
    commit_ivk_chip: CommitIvkChip,
    mut layouter: impl Layouter<pallas::Base>,
    label: &str,
    ak: AssignedCell<pallas::Base, pallas::Base>,
    nk: AssignedCell<pallas::Base, pallas::Base>,
    rivk: ScalarFixed<pallas::Affine, EccChip<OrchardFixedBases>>,
    g_d: &NonIdentityPoint<pallas::Affine, EccChip<OrchardFixedBases>>,
    pk_d_claimed: &NonIdentityPoint<pallas::Affine, EccChip<OrchardFixedBases>>,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, Error> {
    use voting_crypto_deps::composition::commit_ivk;

    let ivk = commit_ivk(
        sinsemilla_chip,
        ecc_chip.clone(),
        commit_ivk_chip,
        layouter.namespace(|| format!("{label} CommitIvk")),
        ak,
        nk,
        rivk,
    )?;

    let ivk_cell = ivk.inner().clone();

    let ivk_scalar = ScalarVar::from_base(
        ecc_chip.clone(),
        layouter.namespace(|| format!("{label} ivk as scalar")),
        ivk.inner(),
    )?;

    let (derived_pk_d, _) = g_d.mul(
        layouter.namespace(|| format!("{label} [ivk] g_d")),
        ivk_scalar,
    )?;

    derived_pk_d.constrain_equal(
        layouter.namespace(|| format!("{label} pk_d equality")),
        pk_d_claimed,
    )?;

    Ok(ivk_cell)
}
