#include <iostream>
#include <cyclus.h>
#include <map>
#include <sqlite_back.h>
#include "prettyprint.hpp"

int main(int argc, char* argv[]) {
  using std::cout;
  std::string table = "Compositions";
  std::string fname = std::string(argv[1]);
  cout << "file name: " << fname << "\n";
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result = fback->Query(table, NULL);
  std::vector<std::string> fld = result.fields;
  cout << fld << std::endl;
  cout << result << std::endl;
  delete fback;
  return 0;
}

