# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-src"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-build"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/tmp"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/src/raylib-populate-stamp"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/src"
  "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/src/raylib-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/src/raylib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/LocalWorkDir/u521785/BMO-Project/BMO Face Engine/build/_deps/raylib-subbuild/raylib-populate-prefix/src/raylib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
