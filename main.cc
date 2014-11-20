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

std::string formatrow(std::vector<std::string>) {
 // must format columns one of these days 
}

int main(int argc, char* argv[]) {
  using std::cout;
  if (argc < 3) {
    cout << "Derp, need at least 2 arguments\n";
  }

  //get and print arguments
  std::string fname = std::string(argv[1]);
  cout << "file name: " << fname << "\n";
  std::string table = std::string(argv[2]);
  cout << "table name: " << table << "\n";
  char const * conds = NULL;
  if (argc > 3) {
    std::vector<cyclus::Cond> conds = std::string(argv[3]);
    cout << "filter conditions: " << conds << "\n";
  }

  //get table from cyclus; print SimId and columns
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result = fback->Query(table, conds);
  cout << "\n" << "SimID: "; 
  cout << result.GetVal<boost::uuids::uuid>("SimId", 0) << "\n\n";
  std::vector<std::string> cols = result.fields;
  std::list<std::string> collist(cols.begin(), cols.end());
  collist.pop_front();

  //print rows of table
  std::vector<cyclus::QueryRow> rows = result.rows;
  cout << collist << "\n"; //  cout << formatrow(cols) << "\n";
  for (int i = 0; i < rows.size(); ++i) {
    std::vector<std::string> stringrow;
    for (int j = 1; j < cols.size(); ++j) {
      std::string s = ToStr(result.types[j], rows[i][j]);
      stringrow.push_back(s);
    }
  cout << stringrow << "\n"; //  cout << formatrow(stringrow) << "\n";  
  }

  delete fback;
  return 0;
}

