#include <iostream>
#include <iomanip>
#include <cyclus.h>
#include <sqlite_back.h>
#include "prettyprint.hpp"

std::string ValToStr(cyclus::DbTypes type, boost::spirit::hold_any val) {
  std::stringstream ss;
  switch(type){
    case cyclus::BOOL:
      ss << val.cast<bool>();
      break;
    case cyclus::INT:
      ss << val.cast<int>();
      break;
    case cyclus::FLOAT:
      ss << val.cast<float>();
      break;
    case cyclus::DOUBLE:
      ss << val.cast<double>();
      break;
    case cyclus::STRING:
      ss << val.cast<std::string>();
      break;
    case cyclus::VL_STRING:
      ss << val.cast<std::string>();
      break;
    case cyclus::BLOB:
      ss << val.cast<cyclus::Blob>().str();
      break;
    case cyclus::UUID:
      ss << val.cast<boost::uuids::uuid>();
      break;
  }
  return ss.str();
}

std::string formatrow(std::vector<std::string>) {
 // must format columns one of these days 
}

cyclus::Cond ParseCond(std::string c) {
  
//  std::string op = OpToStr();
  size_t i = c.find("<");
  std::string field = c.substr(0, i);
  int value = atoi(c.substr(i+1).c_str());
  cyclus::Cond cond = cyclus::Cond(field, std::string("<"), value);
  std::cout << "filter conditions: " << field << " < " << value << "\n";
  return cond;
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
  std::vector<cyclus::Cond> conds;
  for (int i = 3; i < argc; ++i) {
    conds.push_back(ParseCond(std::string(argv[i])));
  }
  
  //get table from cyclus; print SimId and columns
  cyclus::FullBackend* fback = new cyclus::SqliteBack(fname);
  cyclus::QueryResult result;
  if (conds.size() == 0) {
    result = fback->Query(table, NULL);
  } else {
    result = fback->Query(table, &conds);
  }
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
      std::string s = ValToStr(result.types[j], rows[i][j]);
      stringrow.push_back(s);
    }
  cout << stringrow << "\n"; //  cout << formatrow(stringrow) << "\n";  
  }

  delete fback;
  return 0;
}

