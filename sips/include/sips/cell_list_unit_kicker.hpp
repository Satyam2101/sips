#ifndef SIPS_CELL_LIST_UNIT_KICKER_HPP
#define SIPS_CELL_LIST_UNIT_KICKER_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_lists.hpp"
#include "hyperalg/vecN.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "utils/noise.hpp"

namespace ha{

/**
 * class which accumulates the energy and gradient one pair interaction at a time
 */
template <typename distance_policy>
class UnitKickAccumulator {
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double> * m_coords;
    const std::vector<double> m_radii;

public:
    /** record the total kick for each particle*/
    std::vector<double> * m_kick; 

    ~UnitKickAccumulator()
    {
    }

    UnitKickAccumulator(
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : m_dist(dist),
          m_radii(radii)
    {
    }

    void reset_data(const std::vector<double> * coords, std::vector<double> * kick) {
        m_coords = coords;
        m_kick = kick;
    }
    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<m_ndim, double> dr;
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), m_coords->data() + xi_off, m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij;
        double radius_sum = 0;
        if(m_radii.size() > 0) {
            radius_sum = m_radii[atom_i] + m_radii[atom_j];
        }

        if (sqrt(r2)<radius_sum){
            for (size_t k = 0; k < m_ndim; ++k) {
                double dr_hat_k = dr[k]/sqrt(r2);
                (*m_kick)[xi_off + k] += dr_hat_k;
                (*m_kick)[xj_off + k] -= dr_hat_k;
            }
        }
    }
};



/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.h.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename distance_policy>
class CellListUnitKicker{
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    const std::vector<double> m_radii;
    ha::CellLists<distance_policy> m_cell_lists;
    std::shared_ptr<distance_policy> m_dist;

    UnitKickAccumulator<distance_policy> m_kAcc;
public:
    ~CellListUnitKicker(){}
    CellListUnitKicker(
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            const bool balance_omp=true)
        : m_radii(radii),
          m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
          m_dist(dist),
          m_kAcc(dist, m_radii){

    }

    double sum_radii(const size_t atom_i, const size_t atom_j) const {
        if(m_radii.size() == 0) {
            return 0;
        } else {
            return m_radii[atom_i] + m_radii[atom_j];
        }
    }

    size_t get_ndim() { return m_ndim; }


    void get_kick(std::vector<double> const & coords, std::vector<double> & kick)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != kick.size()) {
            throw std::invalid_argument("the kick vector has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(kick.begin(), kick.end(), NAN);
            return;
        }

        update_iterator(coords);
        std::fill(kick.begin(), kick.end(), 0.);
        m_kAcc.reset_data(&coords, &kick);
        auto looper = m_cell_lists.get_atom_pair_looper(m_kAcc);

        looper.loop_through_atom_pairs();
    }

    void get_neighbors(std::vector<double> const & coords,
                                std::vector<std::vector<size_t>> & neighbor_indss,
                                std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                const double cutoff_factor = 1.0)
    {
        size_t natoms = coords.size() / m_ndim;
        std::vector<short> include_atoms(natoms, 1);
        get_neighbors_picky(coords, neighbor_indss, neighbor_distss, include_atoms, cutoff_factor);
    }

    void get_neighbors_picky(std::vector<double> const & coords,
                                      std::vector<std::vector<size_t>> & neighbor_indss,
                                      std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                      std::vector<short> const & include_atoms,
                                      const double cutoff_factor = 1.0)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (natoms != include_atoms.size()) {
            throw std::runtime_error("include_atoms.size() is not equal to the number of atoms");
        }
        if (m_radii.size() == 0) {
            throw std::runtime_error("Can't calculate neighbors, because the "
                                     "used interaction doesn't use radii. ");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return;
        }

        update_iterator(coords);
        ha::NeighborAccumulator<distance_policy> accumulator(m_dist, coords, m_radii, (1) * cutoff_factor, include_atoms);
        auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

        looper.loop_through_atom_pairs();

        neighbor_indss = accumulator.m_neighbor_indss;
        neighbor_distss = accumulator.m_neighbor_distss;
    }

    double get_average_radius(){
        double r_mean = 0.0;
        #ifdef _OPENMP
        #pragma parallel for reduction(+:r_mean)
        for (size_t i = 0;i<m_radii.size();i++){
            r_mean += m_radii[i];
        }
        r_mean /= this->m_natoms;
        #else
        for (size_t i = 0;i<m_radii.size();i++){
            r_mean += m_radii[i];
        }
        r_mean /= this->m_natoms;
        #endif
        return r_mean;
    }

protected:
    virtual void update_iterator(std::vector<double> const & coords)
    {
        m_cell_lists.update(coords);
    }
};

} //namespace ha

#endif 
