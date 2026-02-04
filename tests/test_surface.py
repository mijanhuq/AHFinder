"""Tests for surface mesh functionality."""

import numpy as np
import pytest
from ahfinder.surface import SurfaceMesh, create_sphere, create_ellipsoid


class TestSurfaceMesh:
    """Tests for SurfaceMesh class."""

    def test_creation(self):
        """Test basic mesh creation."""
        mesh = SurfaceMesh(N_s=33)
        assert mesh.N_s == 33
        assert len(mesh.theta) == 33
        assert len(mesh.phi) == 33

    def test_theta_range(self):
        """Test that theta covers [0, π]."""
        mesh = SurfaceMesh(N_s=33)
        assert mesh.theta[0] == 0.0
        assert np.isclose(mesh.theta[-1], np.pi)

    def test_phi_range(self):
        """Test that phi covers [0, 2π)."""
        mesh = SurfaceMesh(N_s=33)
        assert mesh.phi[0] == 0.0
        assert mesh.phi[-1] < 2 * np.pi

    def test_independent_points(self):
        """Test independent point count: N_s² - 2N_s + 2."""
        mesh = SurfaceMesh(N_s=33)
        expected = 33**2 - 2 * 33 + 2
        assert mesh.n_independent == expected

    def test_independent_indices(self):
        """Test that independent indices are computed correctly."""
        mesh = SurfaceMesh(N_s=5)
        indices = mesh.independent_indices()

        # Should have N_s² - 2N_s + 2 = 25 - 10 + 2 = 17 points
        assert len(indices) == 17

        # Poles should only have φ=0
        north_pole = [i for i in indices if i[0] == 0]
        south_pole = [i for i in indices if i[0] == 4]
        assert len(north_pole) == 1
        assert len(south_pole) == 1
        assert north_pole[0][1] == 0
        assert south_pole[0][1] == 0

    def test_xyz_from_rho_sphere(self):
        """Test Cartesian coordinate conversion for a sphere."""
        mesh = SurfaceMesh(N_s=17)
        radius = 2.0
        rho = create_sphere(mesh, radius)
        x, y, z = mesh.xyz_from_rho(rho)

        # All points should be at distance radius from origin
        r = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(r, radius, rtol=1e-10)

    def test_xyz_from_rho_with_center(self):
        """Test coordinate conversion with non-zero center."""
        mesh = SurfaceMesh(N_s=17)
        radius = 1.0
        center = (1.0, 2.0, 3.0)
        rho = create_sphere(mesh, radius)
        x, y, z = mesh.xyz_from_rho(rho, center)

        # All points should be at distance radius from center
        r = np.sqrt((x - 1)**2 + (y - 2)**2 + (z - 3)**2)
        np.testing.assert_allclose(r, radius, rtol=1e-10)

    def test_flat_to_grid_round_trip(self):
        """Test that flat_to_grid and grid_to_flat are inverses."""
        mesh = SurfaceMesh(N_s=17)
        rho = create_sphere(mesh, 2.0)

        flat = mesh.grid_to_flat(rho)
        reconstructed = mesh.flat_to_grid(flat)

        np.testing.assert_allclose(rho, reconstructed, rtol=1e-10)

    def test_pole_replication(self):
        """Test that pole values are replicated correctly."""
        mesh = SurfaceMesh(N_s=9)
        flat = np.random.rand(mesh.n_independent)
        grid = mesh.flat_to_grid(flat)

        # North pole: all φ values should be equal
        assert np.all(grid[0, :] == grid[0, 0])

        # South pole: all φ values should be equal
        assert np.all(grid[-1, :] == grid[-1, 0])


class TestShapeFunctions:
    """Tests for shape creation functions."""

    def test_create_sphere(self):
        """Test sphere creation."""
        mesh = SurfaceMesh(N_s=17)
        radius = 3.0
        rho = create_sphere(mesh, radius)

        assert rho.shape == (17, 17)
        np.testing.assert_allclose(rho, radius)

    def test_create_ellipsoid_sphere(self):
        """Test ellipsoid with equal axes is a sphere."""
        mesh = SurfaceMesh(N_s=17)
        rho = create_ellipsoid(mesh, 2.0, 2.0, 2.0)

        np.testing.assert_allclose(rho, 2.0, rtol=1e-10)

    def test_create_ellipsoid_oblate(self):
        """Test oblate ellipsoid (c < a = b)."""
        mesh = SurfaceMesh(N_s=33)
        a, b, c = 2.0, 2.0, 1.0
        rho = create_ellipsoid(mesh, a, b, c)

        # Equatorial radius should be a
        i_eq = mesh.N_s // 2
        np.testing.assert_allclose(rho[i_eq, :], a, rtol=1e-10)

        # Polar radius should be c
        np.testing.assert_allclose(rho[0, 0], c, rtol=1e-10)
        np.testing.assert_allclose(rho[-1, 0], c, rtol=1e-10)

    def test_create_ellipsoid_prolate(self):
        """Test prolate ellipsoid (c > a = b)."""
        mesh = SurfaceMesh(N_s=33)
        a, b, c = 1.0, 1.0, 2.0
        rho = create_ellipsoid(mesh, a, b, c)

        # Equatorial radius should be a
        i_eq = mesh.N_s // 2
        np.testing.assert_allclose(rho[i_eq, :], a, rtol=1e-10)

        # Polar radius should be c
        np.testing.assert_allclose(rho[0, 0], c, rtol=1e-10)


class TestMeshSpacing:
    """Tests for mesh spacing properties."""

    def test_d_theta(self):
        """Test theta spacing."""
        mesh = SurfaceMesh(N_s=33)
        expected = np.pi / 32
        assert np.isclose(mesh.d_theta, expected)

    def test_d_phi(self):
        """Test phi spacing."""
        mesh = SurfaceMesh(N_s=33)
        expected = 2 * np.pi / 33
        assert np.isclose(mesh.d_phi, expected)

    def test_theta_uniform(self):
        """Test that theta points are uniformly spaced."""
        mesh = SurfaceMesh(N_s=33)
        diffs = np.diff(mesh.theta)
        np.testing.assert_allclose(diffs, mesh.d_theta)

    def test_phi_uniform(self):
        """Test that phi points are uniformly spaced."""
        mesh = SurfaceMesh(N_s=33)
        diffs = np.diff(mesh.phi)
        np.testing.assert_allclose(diffs, mesh.d_phi)
