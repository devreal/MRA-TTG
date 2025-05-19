#ifndef MRA_OPTIONS_H
#define MRA_OPTIONS_H

#include <string>
#include <algorithm>
#include <string_view>

namespace mra {

  struct OptionParser {

  private:
    char **m_begin;
    char **m_end;


    static inline const char *empty = "";

  public:
    OptionParser(int argc, char **argv)
    : m_begin(argv), m_end(argv+argc)
    { }

    std::string_view get(const std::string &option) {
    char **itr = std::find(m_begin, m_end, option);
    if (itr != m_end && ++itr != m_end) return std::string_view(*itr);
      return std::string_view(empty);
    }

    bool exists(const std::string &option) {
      return std::find(m_begin, m_end, option) != m_end;
    }

    int index(const std::string &option) {
      char **itr = std::find(m_begin, m_end, option);
      if (itr != m_end) return (int)(itr - m_end);
      return -1;
    }

    int parse(std::string_view option, int default_value) {
      std::string token;
      int N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stoi(*itr);
      }
      return N;
    }

    long parse(std::string_view option, long default_value) {
      std::string token;
      long N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stol(*itr);
      }
      return N;
    }

    double parse(std::string_view option, double default_value = 0.25) {
      std::string token;
      double N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stod(*itr);
      }
      return N;
    }

  }; // struct Options
} // namespace mra

#endif // MRA_OPTIONS_H