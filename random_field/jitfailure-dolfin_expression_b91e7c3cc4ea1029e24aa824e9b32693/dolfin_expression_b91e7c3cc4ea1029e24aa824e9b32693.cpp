
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_b91e7c3cc4ea1029e24aa824e9b32693 : public Expression
  {
     public:
       

       dolfin_expression_b91e7c3cc4ea1029e24aa824e9b32693()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = y[1] /pow(1,alpha)*sin(1*pi*x[0])+y[2] /pow(2,alpha)*sin(2*pi*x[0])+y[3] /pow(3,alpha)*sin(3*pi*x[0])+y[4] /pow(4,alpha)*sin(4*pi*x[0])+y[5] /pow(5,alpha)*sin(5*pi*x[0])+y[6] /pow(6,alpha)*sin(6*pi*x[0])+y[7] /pow(7,alpha)*sin(7*pi*x[0])+y[8] /pow(8,alpha)*sin(8*pi*x[0])+y[9] /pow(9,alpha)*sin(9*pi*x[0])+y[10] /pow(10,alpha)*sin(10*pi*x[0])+y[11] /pow(11,alpha)*sin(11*pi*x[0]);

       }

       void set_property(std::string name, double _value) override
       {

       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {

       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_b91e7c3cc4ea1029e24aa824e9b32693()
{
  return new dolfin::dolfin_expression_b91e7c3cc4ea1029e24aa824e9b32693;
}

