#ifndef _HYPERALG_ACCUMULATORS_HPP
#define _HYPERALG_ACCUMULATORS_HPP
#include "hyperalg/distance.hpp"
#include "hyperalg/vecN.hpp"

/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename distance_policy>
class NeighborAccumulator {
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> m_coords;
    const std::vector<double> m_radii;
    const double m_cutoff_sca;
    const std::vector<short> m_include_atoms;

public:
    std::vector<std::vector<size_t>> m_neighbor_indss;
    std::vector<std::vector<std::vector<double>>> m_neighbor_distss;

    NeighborAccumulator(
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & coords,
            std::vector<double> const & radii,
            const double cutoff_sca,
            std::vector<short> const & include_atoms)
        : m_dist(dist),
          m_coords(coords),
          m_radii(radii),
          m_cutoff_sca(cutoff_sca),
          m_include_atoms(include_atoms),
          m_neighbor_indss(radii.size()),
          m_neighbor_distss(radii.size())
    {}

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        if (m_include_atoms[atom_i] && m_include_atoms[atom_j]) {
            std::vector<double> dr(m_ndim);
            std::vector<double> neg_dr(m_ndim);
            const size_t xi_off = m_ndim * atom_i;
            const size_t xj_off = m_ndim * atom_j;
            m_dist->get_rij(dr.data(), m_coords.data() + xi_off, m_coords.data() + xj_off);
            double r2 = 0;
            for (size_t k = 0; k < m_ndim; ++k) {
                r2 += dr[k] * dr[k];
                neg_dr[k] = -dr[k];
            }
            const double radius_sum = m_radii[atom_i] + m_radii[atom_j];
            const double r_S = m_cutoff_sca * radius_sum;
            if(sqrt(r2) < r_S) {
                m_neighbor_indss[atom_i].push_back(atom_j);
                m_neighbor_indss[atom_j].push_back(atom_i);
                m_neighbor_distss[atom_i].push_back(dr);
                m_neighbor_distss[atom_j].push_back(neg_dr);
            }
        }
    }
};
#endif