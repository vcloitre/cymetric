#include <iostream>
#include <iomanip>
#include <cyclus.h>
#include <sqlite_back.h>
#include "prettyprint.hpp"

//ValToStr converts any data type to a string for printing 
std::string ValToStr(boost::spirit::hold_any val, cyclus::DbTypes type) {
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

//formatrow pretty prints the table
std::string formatrow(std::vector<std::string>) {
 // must format columns one of these days 
}

//TypeConv converts the input string value to the relevant data type
boost::spirit::hold_any TypeConv(std::string valstr, std::string field){
  boost::spirit::hold_any val;
  if (field == "NucId") {
    val = strtol(valstr.c_str(), NULL, 10);
  } else if (field == "MassFrac") {
    val = atof(valstr.c_str());
  } else {
    std::cout << "Derp, data type not supported\n";
  }
  return val;
}

//prints a condition
void PrintCond(std::string field, std::string op, std::string valstr){
  std::cout << "filter conditions: " << field << " " << op  << " " << valstr << "\n";
}

//ParseCond separates the conditions string for formatting 
cyclus::Cond ParseCond(std::string c) {
  std::vector<std::string> ops = {"<", ">", "<=", ">=", "==", "!="};

  //finds the logical operator in the string
  std::string op;
  bool exists = false;
  for (int i = 0; !exists; ++i) {
    op = ops[i];
    exists = c.find(op) != std::string::npos; 
  }

  //finds the location of the logical operator
  size_t i = c.find(op);

  //gives substrings separated by the location of the operator
  std::string field = c.substr(0, i);
  char* cop = (char*)op.c_str(); 
  size_t j = strlen(cop);
  boost::spirit::hold_any value;
  std::string valstr;
  if (j == 2) {
    valstr = c.substr(i+2);
    value = TypeConv(valstr, field);
  } else {
    valstr = c.substr(i+1);
    value = TypeConv(valstr, field);
  }

  //populates cyclus-relevant condition
  cyclus::Cond cond = cyclus::Cond(field, op, value);
  PrintCond(field, op, valstr);
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
      std::string s = ValToStr(rows[i][j], result.types[j]);
      stringrow.push_back(s);
    }
  cout << stringrow << "\n"; //  cout << formatrow(stringrow) << "\n";  
  }

  delete fback;
  return 0;
}
