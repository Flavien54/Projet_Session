"""Tests unitaires de calcul_Xfoil (helpers et worker).

Couvre le filtrage physique des coefficients, le nettoyage des
aberrations, l'extrapolation bornée et la routine worker complète
avec XFoil remplacé par un stub.
"""

import numpy as np
import pytest

import calcul_Xfoil as cx


# ── _is_physical ─────────────────────────────────────────────────

class TestIsPhysical:
    def test_valeurs_normales(self):
        assert cx._is_physical(0.8, 0.02, -0.05) is True

    def test_cl_hors_bornes(self):
        assert cx._is_physical(4.0, 0.02, 0.0) is False    # > 3.5
        assert cx._is_physical(-3.0, 0.02, 0.0) is False   # < -2.5

    def test_cd_hors_bornes(self):
        assert cx._is_physical(0.5, 0.0, 0.0) is False     # < 1e-5
        assert cx._is_physical(0.5, 0.6, 0.0) is False     # > 0.5

    def test_cm_hors_bornes(self):
        assert cx._is_physical(0.5, 0.02, 1.5) is False
        assert cx._is_physical(0.5, 0.02, -1.5) is False

    def test_bornes_incluses(self):
        assert cx._is_physical(*cx.CL_BOUNDS[:1], 0.02, 0.0) is True
        assert cx._is_physical(0.5, cx.CD_BOUNDS[0], 0.0) is True


# ── _sanitize_map ────────────────────────────────────────────────

class TestSanitizeMap:
    def test_filtre_les_aberrations(self):
        brut = {
            0.0: (0.5, 0.02, -0.05),     # valide
            1.0: (9.9, 0.02, 0.0),       # CL aberrant
            2.0: (0.7, 0.0, 0.0),        # CD nul -> rejeté
        }
        propre = cx._sanitize_map(brut)
        assert list(propre) == [0.0]

    def test_map_vide(self):
        assert cx._sanitize_map({}) == {}


# ── _extrapolate ─────────────────────────────────────────────────

class TestExtrapolate:
    ALPHAS = np.arange(-2.0, 3.0, 1.0)  # -2, -1, 0, 1, 2

    def test_map_vide_donne_zeros(self):
        resultat = cx._extrapolate(self.ALPHAS, {})
        assert len(resultat) == len(self.ALPHAS)
        assert all(v == (0.0, 0.0, 0.0) for v in resultat.values())

    def test_couverture_complete(self):
        connu = {0.0: (0.5, 0.02, -0.05), 2.0: (0.9, 0.03, -0.06)}
        resultat = cx._extrapolate(self.ALPHAS, connu)
        for a in self.ALPHAS:
            assert round(float(a), 2) in resultat

    def test_points_connus_inchanges(self):
        connu = {0.0: (0.5, 0.02, -0.05), 2.0: (0.9, 0.03, -0.06)}
        resultat = cx._extrapolate(self.ALPHAS, connu)
        assert resultat[0.0] == connu[0.0]
        assert resultat[2.0] == connu[2.0]

    def test_interpolation_lineaire(self):
        connu = {0.0: (0.0, 0.01, 0.0), 2.0: (1.0, 0.03, -0.1)}
        resultat = cx._extrapolate(self.ALPHAS, connu)
        cl, cd, cm = resultat[1.0]
        assert cl == pytest.approx(0.5)
        assert cd == pytest.approx(0.02)
        assert cm == pytest.approx(-0.05)

    def test_extrapolation_plate_aux_bords(self):
        # np.interp bloque aux valeurs limites (flat-clamp) : pas d'explosion
        connu = {0.0: (0.5, 0.02, -0.05), 1.0: (0.6, 0.025, -0.06)}
        resultat = cx._extrapolate(self.ALPHAS, connu)
        assert resultat[-2.0] == pytest.approx((0.5, 0.02, -0.05))
        assert resultat[2.0] == pytest.approx((0.6, 0.025, -0.06))

    def test_valeurs_dans_les_bornes_physiques(self):
        connu = {0.0: (3.5, 0.5, 1.0)}  # aux limites
        resultat = cx._extrapolate(self.ALPHAS, connu)
        for cl, cd, cm in resultat.values():
            assert cx.CL_BOUNDS[0] <= cl <= cx.CL_BOUNDS[1]
            assert cx.CD_BOUNDS[0] <= cd <= cx.CD_BOUNDS[1]
            assert cx.CM_BOUNDS[0] <= cm <= cx.CM_BOUNDS[1]


# ── compute_profile_re (worker complet, XFoil stubé) ─────────────

GEOM_ROW = {
    "source": "naca_grid", "t": 0.12, "camber": 0.02, "x_t": 0.30,
    "x_c": 0.40, "LE_radius": 0.015, "TE_angle": 12.0,
    "t_over_xt": 0.40, "area": 0.08,
}


class TestComputeProfileRe:
    def test_succes_avec_convergence_complete(self, monkeypatch):
        alphas = np.arange(0.0, 5.0, 1.0)

        def xfoil_stub(airfoil, re_val, seq, max_iter, timeout):
            return {round(float(a), 2): (0.1 * a, 0.01, -0.02) for a in seq}

        monkeypatch.setattr(cx, "_run_xfoil", xfoil_stub)
        res = cx.compute_profile_re(("naca2412", 1e5, alphas, GEOM_ROW))

        assert res["success"] is True
        assert res["n_conv"] == len(alphas)
        assert len(res["rows"]) == len(alphas)
        assert all(r["converged"] for r in res["rows"])

    def test_ld_calcule_et_borne(self, monkeypatch):
        alphas = np.array([0.0])

        def xfoil_stub(airfoil, re_val, seq, max_iter, timeout):
            # CD minuscule -> L/D brut >> LD_MAX, doit être borné
            return {0.0: (3.0, 1e-4, 0.0)}

        monkeypatch.setattr(cx, "_run_xfoil", xfoil_stub)
        res = cx.compute_profile_re(("naca2412", 1e5, alphas, GEOM_ROW))
        assert res["rows"][0]["LD"] == cx.LD_MAX

    def test_non_convergence_extrapolee(self, monkeypatch):
        alphas = np.arange(0.0, 5.0, 1.0)

        def xfoil_stub(airfoil, re_val, seq, max_iter, timeout):
            # Seul alpha=0 converge, tout le reste sera interpolé
            return {0.0: (0.5, 0.02, -0.05)} if 0.0 in np.round(seq, 2) else {}

        monkeypatch.setattr(cx, "_run_xfoil", xfoil_stub)
        res = cx.compute_profile_re(("naca2412", 1e5, alphas, GEOM_ROW))

        assert res["success"] is True
        assert res["n_conv"] == 1
        assert len(res["rows"]) == len(alphas)   # aucun trou
        interpoles = [r for r in res["rows"] if not r["converged"]]
        assert len(interpoles) == len(alphas) - 1

    def test_echec_total_retourne_erreur(self, monkeypatch):
        def airfoil_casse(name):
            raise RuntimeError("profil introuvable")

        monkeypatch.setattr(cx.asb, "Airfoil", airfoil_casse)
        res = cx.compute_profile_re(("inconnu", 1e5, np.array([0.0]), GEOM_ROW))
        assert res["success"] is False
        assert res["rows"] == []
        assert "profil introuvable" in res["error"]
