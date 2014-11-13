#include <iostream>
#include <cyclus.h>
#include <sqlite_back.h>
#include "prettyprint.hpp"

int main(int argc, char* argv[]) {
  using std::cout;
  std::string table = "Compositions";
  std::string fname = std::string(argv[1]);
  cout << "file name: " << fname << "\n";
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result = fback->Query(table, NULL);
  for (int i = 0; i < result.rows.size(); ++i) {
    cout << result.GetVal<std::string>("SimID", i) << "\n";
    cout << result.GetVal<int>("QualID", i) << "\n";
    cout << result.GetVal<int>("NucID", i) << "\n";
    cout << result.GetVal<double>("MassFrac", i) << "\n";
  }
  delete fback;
  return 0;
}

