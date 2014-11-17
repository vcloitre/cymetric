#include <iostream>
#include <iomanip>
#include <cyclus.h>
#include <sqlite_back.h>
#include "prettyprint.hpp"

std::string ToStr(cyclus::DbTypes type, boost::spirit::hold_any val) {
  std::stringstream ss;
  switch(type){
    case 0:
      ss << val.cast<bool>();
      break;
    case 1:
      ss << val.cast<int>();
      break;
    case 2:
      ss << val.cast<float>();
      break;
    case 3:
      ss << val.cast<double>();
      break;
    case 4:
      ss << val.cast<std::string>();
      break;
    case 5:
      ss << val.cast<std::string>();
      break;
//    case 6:
//      ss << val.cast<cyclus::Blob>();
//      break;
    case 7:
      ss << val.cast<boost::uuids::uuid>();
      break;
  }
  return ss.str();
}

int main(int argc, char* argv[]) {
  using std::cout;
  std::string fname = std::string(argv[1]);
  std::string table = std::string(argv[2]);
  cout << "file name: " << fname << "\n";
  cout << "table name: " << table << "\n";
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result = fback->Query(table, NULL);
  cout << "\n" << "SimID: "; 
  cout << result.GetVal<boost::uuids::uuid>("SimId", 0) << "\n\n";
  std::vector<std::string> cols = result.fields;
  std::vector<cyclus::QueryRow> lines = result.rows;
  for (int i = 0; i < lines.size(); ++i) {
    cout << lines << "\n";

//    for (int j = 0; j < cols.size(); ++j) {
//     cyclus::DbTypes type = result.types[j];
//      boost::spirit::hold_any val = "";
//      std::string dtype = ToStr(type, val);
//      cout << result.GetVal<dtype>(cols[j], i);
//    }
  }
  delete fback;
  return 0;
}

