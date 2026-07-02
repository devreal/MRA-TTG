#ifndef MRA_MISC_INIT_H
#define MRA_MISC_INIT_H


namespace mra {

  /**
   * Initializes the runtime systems necessary for MRA.
   * This includes TTG, MADNESS, and some adjustments to TA.
   * TODO: take a MADNESS world as argument.
   */
  void initialize(int& argc, char **& argv, int ncores);

  /**
   * Finalize the environment set up in mra::initialize.
   */
  void finalize();


} // namespace mra


#endif // MRA_MISC_INIT_H