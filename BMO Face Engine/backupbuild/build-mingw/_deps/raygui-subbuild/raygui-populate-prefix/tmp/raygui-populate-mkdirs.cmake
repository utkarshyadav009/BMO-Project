# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-src")
  file(MAKE_DIRECTORY "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-build"
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix"
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/tmp"
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/src/raygui-populate-stamp"
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/src"
  "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/src/raygui-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/src/raygui-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/raouy/OneDrive/Documents/GitHub/BMO Project/BMO-Project/BMO Face Engine/build-mingw/_deps/raygui-subbuild/raygui-populate-prefix/src/raygui-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
