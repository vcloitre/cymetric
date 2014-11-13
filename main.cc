#include <iostream>
#include <iomanip>
#include <cyclus.h>
#include <sqlite_back.h>
#include "prettyprint.hpp"

void dbtype(cyclus::DbTypes t, void* x) {
  switch(t){
    case "BOOL":
      (bool*)x;
      break;
    case "INT":
      (int*)x;
      break;
    case "FLOAT":
      (float*)x;
      break;
    case "DOUBLE":
      (double*)x;
      break;
    case "STRING":
      (std::string*)x;
      break;
    case "VL_STRING":
      (cyclus::vl_string*)x;
      break;
    case "BLOB":
      (cyclus::blob*)x;
      break;
    case "UUID":
      (boost::uuids::uuid*)x;
      break;
  }
}

int main(int argc, char* argv[]) {
  using std::cout;
  std::string fname = std::string(argv[1]);
  std::string table = std::string(argv[2]);
  cout << "file name: " << fname << "\n";
  cout << "table name: " << table << "\n";
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result = fback->Query(table, NULL);
  cout << "\n" << "SimID: " << result.GetVal<boost::uuids::uuid>("SimId", 0) << "\n\n";
  std::vector<std::string> cols = result.fields;
  for (int i = 0; i < result.rows.size(); ++i) {
//    cout << result.GetVal<int>(cols[2], i);
    for (int j = 0; i < cols.size(); ++j) {
      int e = result.types[j];
      void dtype = dbtype(e);
      cout << result.GetVal<dtype>(cols[j], i);
    }
  }
  delete fback;
  return 0;
}

