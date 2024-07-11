#ifndef FILE_HELPER_HPP
#define FILE_HELPER_HPP
#include <cstdlib>
#include <string>
#include <stdexcept>

std::string path_join(std::string dir, std::string file){
    if (dir.back() != '/') {
        return dir + '/' + file;
    }
    else{
        return dir + file;
    }
}

void compress_file(std::string dir,std::string target, std::string source){
    target = path_join(dir,target);
    source = path_join(dir,source);
    std::string cmd = "zip -j " + target + " " + source;
    if (std::system(cmd.c_str()) != 0){
        throw std::runtime_error("cannot compress file");
    }
}

void rm_file(std::string dir, std::string file){
    std::string cmd = "rm " + path_join(dir,file);
    if (std::system(cmd.c_str()) != 0){
        throw std::runtime_error("cannot compress file");
    }
}

#endif