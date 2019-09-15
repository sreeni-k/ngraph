//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <fstream>
#include <iomanip>

#include "json2ngraph.hpp"
#include "ngraph/file_util.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    string input;
    string output;
    bool failed = true;
    int rc = -1;

    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-i" || arg == "--input")
        {
            input = argv[++i];
        }
        else if (arg == "-o" || arg == "--output")
        {
            output = argv[++i];
        }
    }
    if (input.empty())
    {
        cerr << "Input file missing\n";
    }
    else
    {
        failed = false;
    }

    if (failed)
    {
        cerr << R"###(DESCRIPTION
    Convert a json serialized model into ngraph c++ code.

SYNOPSIS
    json2ngraph [-i|--input <input filename>] [-o|--output <output file>]

OPTIONS
        -i|--input      Path to input file
        -o|--output     Path to output file
)###";
        return 1;
    }

    ifstream in(input);
    if (in)
    {
        json2ngraph(in, cout);
    }
    else
    {
        cerr << "Failed to open input file '" << input << "'\n";
    }

    return rc;
}
