# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Common MLIR/LLVM setup for the sub-directories below

message(STATUS "LLVM Directory: ${LLVM_DIR}")
# Link LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
set(MLIR_LLVM_INCLUDE_PATH ${LLVM_INCLUDE_DIRS})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

message(STATUS "MLIR Headers at : ${MLIR_INCLUDE_PATHS}")
message(STATUS "LLVM Headers at : ${MLIR_LLVM_INCLUDE_PATH}")

include_directories(${NGRAPH_INCLUDE_PATH};${MLIR_LLVM_INCLUDE_PATH};${MLIR_INCLUDE_PATHS}) 
 
add_subdirectory(compiler)
add_subdirectory(tools/ngraph-opt)
