#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from argparse import ArgumentParser
from pathlib import Path

from autopxd import parse
from autopxd.nodes import Block
from autopxd.writer import AutoPxd, escape
from pycparser import c_ast


class OvmsAutoPxd(AutoPxd):
    apply_cimport_bool_wa = False   # apply WA if any 'bool' type was used in header file.

    def visit_IdentifierType(self, node):
        super().visit_IdentifierType(node)
        if node.names[0] == 'bool':
            self.apply_cimport_bool_wa = True

    def visit_Struct(self, node):
        kind = "struct"
        if "OVMS" in node.name and node.decls is None:
            name = node.name
            fields = []
            type_decl = self.child_of(c_ast.TypeDecl, -2)
            # add the struct definition to the top level
            self.decl_stack[0].append(Block(escape(name, True), fields, kind, "cdef"))
            if type_decl:
                # inline struct, add a reference to whatever name it was defined on the top level
                self.append(escape(name))
        else:
            return self.visit_Block(node, kind)

    def translate(self, code):
        self.visit(parse(code=code))
        pxd_string = ""
        if self.stdint_declarations:
            cimports = ", ".join(self.stdint_declarations)
            pxd_string += f"from libc.stdint cimport {cimports}\n\n"
            if self.apply_cimport_bool_wa:
                # Workaround for cython issue: 'bool' is not a type identifier
                # https://stackoverflow.com/questions/24659723/cython-issue-bool-is-not-a-type-identifier
                pxd_string += "from libcpp cimport bool\n\n"
        pxd_string += str(self)
        return pxd_string


if __name__ == "__main__":
    parser = ArgumentParser(description="Script translates OVMS header file to .pxd file")
    parser.add_argument("-i", "--input_file", help="OVMS header file path")
    parser.add_argument("-o", "--output_file", help=".pxd output file path")
    
    args = parser.parse_args()

    if len(sys.argv) !=5:
        args = parser.parse_args(["-h"])

    input_file_path = Path(args.input_file)
    output_file_path = Path(args.output_file)

    with open(output_file_path, "w") as fo:
        fo.write(OvmsAutoPxd(input_file_path.name).translate(input_file_path.read_text()))
