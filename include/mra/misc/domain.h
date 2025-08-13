#ifndef MADDOMAIN_H_INCL
#define MADDOMAIN_H_INCL

#include <array>
#include <iostream>
#include <utility>
#include <cmath>
#include <cassert>

#include "mra/misc/types.h"
#include "mra/misc/key.h"
#include "mra/misc/platform.h"

namespace mra {

    template <Dimension NDIM>
    struct Domain {
        // Rather than this static data we might be better off with a singleton object that can be easily serialized/communicated, etc. ?

        // Also, might want to associate functions with a domain to ensure consistency?

        // Eliminated storing both double and float data since the data is accessed outside of inner loops

        // Also, the C++ parser requires Domain<NDIM>:: template get<T>(d) ... ugh!
    private:
        struct Range {
            double lo, hi;
        };
        std::array<Range,NDIM>  cell;
        std::array<double,NDIM> cell_width;
        std::array<double,NDIM> cell_reciprocal_width;
        double cell_volume;
        bool initialized;

        void set(Dimension d, double lo, double hi) {
            assert(d<NDIM);
            assert(hi>lo);

            cell[d] = {lo,hi};
            cell_width[d] = hi - lo;
            cell_reciprocal_width[d] = 1.0/cell_width[d];
            cell_volume = 1.0;
            for (double x : cell_width) cell_volume *= x;
        }

    public:

        Domain() = default;
        Domain(Domain&& domain) = delete;
        Domain(const Domain& domain) = default;
        Domain& operator=(Domain&& domain) = delete;
        Domain& operator=(const Domain& domain) = default;


        void set_cube(double lo, double hi) {
            for (Dimension d = 0; d < NDIM; ++d) set(d, lo, hi);
            initialized = true;
        }

        /// Returns the simulation domain in dimension d as a pair of values (first=lo, second=hi)
        SCOPE std::pair<double,double> get(Dimension d) const {
            assert(d<NDIM);
            assert(initialized);
            return std::make_pair(cell[d].lo, cell[d].hi);
        }

        /// Returns the width of dimension d
        SCOPE double get_width(Dimension d) const {
            assert(d<NDIM);
            assert(initialized);
            return cell_width[d];
        }

        /// Returns the maximum width of any dimension
        SCOPE double get_max_width() const {
            double w = 0.0;
            for (Dimension d = 0; d < NDIM; ++d) w = std::max(w,get_width(d));
            return w;
        }
        template <typename T>
        SCOPE double get_reciprocal_width(Dimension d) const {
            assert(d<NDIM);
            assert(initialized);
            return cell_reciprocal_width[d];
        }

        template <typename T>
        SCOPE T get_volume() const {
            assert(initialized);
            return cell_volume;
        }

        /// Convert user coords (Domain) to simulation coords ([0,1]^NDIM)
        template <typename T>
        SCOPE void user_to_sim(const Coordinate<T,NDIM>& xuser, Coordinate<T,NDIM>& xsim) const {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            for (Dimension d=0; d<NDIM; ++d)
                xsim[d] = (xuser[d] - cell[d].lo) * cell_reciprocal_width[d];
            //return xsim;
        }

        /// Convert simulation coords ([0,1]^NDIM) to user coords (Domain)
        template <typename T>
        SCOPE void sim_to_user(const Coordinate<T,NDIM>& xsim, Coordinate<T,NDIM>& xuser) const {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            for (Dimension d=0; d<NDIM; ++d) {
                xuser[d] = xsim[d]*cell_width[d] + cell[d].lo;
            }
        }

        /// Returns the corners in user coordinates (Domain) that bound the box labelled by the key
        template <typename T>
        SCOPE std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>> bounding_box(const Key<NDIM>& key) const {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            Coordinate<T,NDIM> lo, hi;
            const T h = std::pow(T(0.5),T(key.level()));
            const std::array<Translation,NDIM>& l = key.translation();
            for (Dimension d=0; d<NDIM; ++d) {
                T box_width = h*cell_width[d];
                lo[d] = cell[d].lo + box_width*l[d];
                hi[d] = lo[d] + box_width;
            }
            return std::make_pair(lo,hi);
        }

        /// Returns the box at level n that contains the given point in simulation coordinates
        /// @param[in] pt point in simulation coordinates
        /// @param[in] n the level of the box
        template <typename T>
        SCOPE Key<NDIM> sim_to_key(Batch b, const Coordinate<T,NDIM>& pt, Level n) const {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            std::array<Translation,NDIM> l;
            T twon = std::pow(T(2.0), T(n));
            for (Dimension d=0; d<NDIM; ++d) {
                l[d] = Translation(twon*pt[d]);
            }
            return Key<NDIM>(b, n,l);
        }

        SCOPE void print() const {
            assert(initialized);
            std::cout << "Domain<" << NDIM << ">(" << Domain<NDIM>::cell << ")\n";
        }
    };
}

#endif
