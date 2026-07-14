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

from tests.functional.constants.ovms import Ovms

SHAPE_AND_LAYOUT_EFFECTS_TABLE = """
      Shape and layout effects table.

Native Model    ; Native Model Layout  ; Model Optimizer       ;OVMS layout parameter  ;Expected OVMS metadata ; Is         ; Is binary resize 
Shape           ; (Inherited from      ; layout parameter      ;                       ;                       ; binary     ; auto alignment 
                ;   training           ;                       ;                       ;                       ; supported  ; supported
                ;   framework)         ;                       ;                       ;                       ;            ;
                ;                      ;                       ;                       ;                       ;            ;
(1,224,224,3)   ;     NHWC             ;                       ;                       ;(1,224,224,3) N...     ;yes         ;   no
(1,224,224,3)   ;     NHWC             ;                       ;--layout NCHW          ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ;                       ;--layout NHWC          ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ;                       ;--layout NCHW:NHWC     ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ;                       ;--layout NHWC:NCHW     ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NHWC         ;                       ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NHWC         ;--layout NCHW          ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NHWC         ;--layout NHWC          ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NHWC         ;--layout NCHW:NHWC     ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NHWC         ;--layout NHWC:NCHW     ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NCHW         ;                       ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NCHW         ;--layout NCHW          ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NCHW         ;--layout NHWC          ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NCHW         ;--layout NCHW:NHWC     ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NCHW         ;--layout NHWC:NCHW     ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NHWC->NCHW   ;                       ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NHWC->NCHW   ;--layout NCHW          ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NHWC->NCHW   ;--layout NHWC          ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NHWC->NCHW   ;--layout NCHW:NHWC     ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NHWC->NCHW   ;--layout NHWC:NCHW     ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NCHW->NHWC   ;                       ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NCHW->NHWC   ;--layout NCHW          ;(1,224,3,224) NCHW     ;no;         n   ./a
(1,224,224,3)   ;     NHWC             ; --layout NCHW->NHWC   ;--layout NHWC          ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,224,224,3)   ;     NHWC             ; --layout NCHW->NHWC   ;--layout NCHW:NHWC     ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,224,224,3)   ;     NHWC             ; --layout NCHW->NHWC   ;--layout NHWC:NCHW     ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ;                       ;                       ;(1,3,224,224) N...     ;yes         ;   no
(1,3,224,224)   ;     NCHW             ;                       ;--layout NCHW          ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ;                       ;--layout NHWC          ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ;                       ;--layout NCHW:NHWC     ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ;                       ;--layout NHWC:NCHW     ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NHWC         ;                       ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NHWC         ;--layout NCHW          ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NHWC         ;--layout NHWC          ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NHWC         ;--layout NCHW:NHWC     ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NHWC         ;--layout NHWC:NCHW     ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NCHW         ;                       ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NCHW         ;--layout NCHW          ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NCHW         ;--layout NHWC          ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NCHW         ;--layout NCHW:NHWC     ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NCHW         ;--layout NHWC:NCHW     ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NHWC->NCHW   ;                       ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NHWC->NCHW   ;--layout NCHW          ;(1,224,3,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NHWC->NCHW   ;--layout NHWC          ;(1,224,3,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NHWC->NCHW   ;--layout NCHW:NHWC     ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NHWC->NCHW   ;--layout NHWC:NCHW     ;(1,3,224,224) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NCHW->NHWC   ;                       ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NCHW->NHWC   ;--layout NCHW          ;(1,224,224,3) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NCHW->NHWC   ;--layout NHWC          ;(1,224,224,3) NHWC     ;yes         ;   yes
(1,3,224,224)   ;     NCHW             ; --layout NCHW->NHWC   ;--layout NCHW:NHWC     ;(1,3,224,224) NCHW     ;no          ;   n/a
(1,3,224,224)   ;     NCHW             ; --layout NCHW->NHWC   ;--layout NHWC:NCHW     ;(1,224,3,224) NHWC     ;yes         ;   yes

"""


class Shape(list):
    layout = None
    mappings = []

    def __init__(self, _list, _layout=None):
        self.init_by_list(_list, _layout)

    def set_layout(self, _list, _layout=None):
        if not _layout:
            # Set default layout
            if len(_list) == 2:
                self.layout = "NC"
            elif len(_list) == 3:
                self.layout = "NCW"
            elif len(_list) == 4:
                self.layout = "NCHW"
            elif len(_list) == 5:
                self.layout = "NCHWD"
            else:
                assert "Unrecognized shape"
        else:
            if _layout in [Ovms.LAYOUT_NHWC, Ovms.LAYOUT_NCHW]:
                _layout = _layout.split(":")[0]
            else:
                assert 0
            self.layout = _layout
            assert len(_list) == len(_layout)

    def init_by_list(self, _list, _layout=None):
        self.set_layout(_list, _layout)
        for i in range(len(_list)):
            setattr(self, self.layout[i], _list[i])
        self[:] = _list[:]

    def get_shape_by_layout(self, layout=None):
        if not layout:
            layout = self.layout
        if layout:
            return [getattr(self, letter) for letter in layout]
        return self[:]
