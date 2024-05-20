#pragma once
#include <gpu-api.h>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

namespace rtat {

struct Problem {
  int m, k, n;
  cublasOperation_t opA, opB;

  Problem() {}

  Problem(int m, int k, int n, 
          cublasOperation_t opA, 
          cublasOperation_t opB) :
      m(m), k(k), n(n), opA(opA), opB(opB) {}

  friend std::ostream& operator<<(std::ostream& os, const Problem &problem) {
    os << problem.m << " " << problem.k << " " << problem.n << " "
       << op_to_char(problem.opA) << " " << op_to_char(problem.opB);
    return os;
  }

  friend std::istream& operator>>(std::istream &is, Problem &problem) {
    is >> problem.m;
    is >> problem.k;
    is >> problem.n;
    char A, B;
    is >> A;
    is >> B;
    problem.opA = char_to_op(A);
    problem.opB = char_to_op(B);

    return is;
  }

  void print() const {
    std::cout << "m=" << m 
              << ", k=" << k 
              << ", n=" << n 
              << ", " << op_to_char(opA) << op_to_char(opB) 
              << std::endl;
  }

private:
  static std::string op_to_char(cublasOperation_t op) {
    if (op == CUBLAS_OP_N) return "N";
    if (op == CUBLAS_OP_T) return "T";
    return "0";
  }

  static cublasOperation_t char_to_op(char c) {
    if (c == 'N') return CUBLAS_OP_N;
    if (c == 'T') return CUBLAS_OP_T;
    return CUBLAS_OP_N;
  }

};

class Problem_Set {
  const std::string filename = "";
  std::vector<Problem> problems;
  std::vector<std::string> comments;

public:
  friend std::ostream& operator<<(std::ostream& os, const Problem_Set &problem_set) {
    for (auto &comment : problem_set.comments)
      os << comment << std::endl;
    for (auto &problem : problem_set.problems)
      os << problem << std::endl;
    return os;
  }

  friend std::istream& operator>>(std::istream& is, Problem_Set &problem_set) {
    std::string s;
    while (is.peek() == '#' && std::getline(is,s)) 
      problem_set.comments.push_back(s);

    while (std::getline(is,s)) {
      std::stringstream ss(s);

      Problem problem;
      ss >> problem;
      if (ss.fail()) {
        std::cout << "Parse failure on line: " << s << ", size " << s.size() << std::endl;
        throw;
      }
      std::cout << "Add problem " << problem << std::endl;

      problem_set.problems.push_back(problem);
    } 
    return is;
  }

  Problem_Set() : filename(""), problems() {}
  Problem_Set(std::string filename) : filename(filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
      std::cout << "Bad filename " << filename << " given to Problem_Set" << std::endl;
      throw;
    }
    is >> *this;
  }

  ~Problem_Set() {
    dump();
  }

  void dump(std::string file) {
    if (file == "") return;

    std::ofstream os(file, std::ios::trunc);
    os << *this;
  }

  void dump() {
    dump(filename);
  }

  void add_problem(Problem p) { problems.push_back(p); }

  bool has_duplicate_dimensions() {
    std::set<std::tuple<int,int,int>> dim_set;
    for (auto &problem : problems)
      dim_set.insert(std::make_tuple(problem.m, problem.k, problem.n));
    return dim_set.size() != problems.size();
  }

  std::vector<Problem>& get_problems() { return problems; } 
};
}
