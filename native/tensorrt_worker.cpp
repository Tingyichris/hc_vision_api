#include <cstdint>
#include <iostream>
#include "runner.hpp"
#include <string>
#include <vector>
int main(int argc,char** argv){if(argc!=2)return 2;Runner runner(argv[1]);std::uint32_t n=0;while(std::cin.read(reinterpret_cast<char*>(&n),4)){std::vector<unsigned char>b(n);if(!std::cin.read(reinterpret_cast<char*>(b.data()),n))break;std::string out=runner.infer(b);auto m=static_cast<std::uint32_t>(out.size());std::cout.write(reinterpret_cast<char*>(&m),4).write(out.data(),m).flush();}return 0;}
