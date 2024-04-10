
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
  class dolfin_expression_78c8528fdde94efb229051e6d061e114 : public Expression
  {
     public:
       std::shared_ptr<dolfin::GenericFunction> generic_function_sampled;


       dolfin_expression_78c8528fdde94efb229051e6d061e114()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          double sampled[500];

            generic_function_sampled->eval(Eigen::Map<Eigen::Matrix<double, 500, 1>>(sampled), x);
          values[0] = ( cos(2 * pi * 1 * x[0] ) + cos(2 * pi * 1 * x[1] ) ) / (1 * 1 * pi * pi) * sampled[0][0] + ( cos(2 * pi * 2 * x[0] ) + cos(2 * pi * 2 * x[1] ) ) / (2 * 2 * pi * pi) * sampled[1][0] + ( cos(2 * pi * 3 * x[0] ) + cos(2 * pi * 3 * x[1] ) ) / (3 * 3 * pi * pi) * sampled[2][0] + ( cos(2 * pi * 4 * x[0] ) + cos(2 * pi * 4 * x[1] ) ) / (4 * 4 * pi * pi) * sampled[3][0] + ( cos(2 * pi * 5 * x[0] ) + cos(2 * pi * 5 * x[1] ) ) / (5 * 5 * pi * pi) * sampled[4][0] + ( cos(2 * pi * 6 * x[0] ) + cos(2 * pi * 6 * x[1] ) ) / (6 * 6 * pi * pi) * sampled[5][0] + ( cos(2 * pi * 7 * x[0] ) + cos(2 * pi * 7 * x[1] ) ) / (7 * 7 * pi * pi) * sampled[6][0] + ( cos(2 * pi * 8 * x[0] ) + cos(2 * pi * 8 * x[1] ) ) / (8 * 8 * pi * pi) * sampled[7][0] + ( cos(2 * pi * 9 * x[0] ) + cos(2 * pi * 9 * x[1] ) ) / (9 * 9 * pi * pi) * sampled[8][0] + ( cos(2 * pi * 10 * x[0] ) + cos(2 * pi * 10 * x[1] ) ) / (10 * 10 * pi * pi) * sampled[9][0];

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
          if (name == "sampled") { generic_function_sampled = _value; return; }
       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {
          if (name == "sampled") return generic_function_sampled;
       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_78c8528fdde94efb229051e6d061e114()
{
  return new dolfin::dolfin_expression_78c8528fdde94efb229051e6d061e114;
}

