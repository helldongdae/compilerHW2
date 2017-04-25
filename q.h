//
//  myscanner.h
//  yylex
//
//  Created by Um HaYoung on 2017. 4. 22..
//  Copyright © 2017년 Um HaYoung. All rights reserved.
//

#ifndef myscanner_h
#define myscanner_h

#include <stdio.h>

#define TYPE_comment        1
#define TYPE_comment_newline 2
#define TYPE_int            3
#define TYPE_float          4
#define TYPE_bool           5
#define TYPE_char           6
#define TYPE_str            7
#define TYPE_general        8
#define TYPE_null           9
#define TYPE_op             10
#define TYPE_separator      11
#define TYPE_charKeyBoard   12
#define TYPE_id             13
#define TYPE_method         14
#define TYPE_newline        15
#define TYPE_space          16
#define TYPE_type	    17


//RGB => (0xFF,0xFF,0xFF)
#define COLOR_background    "000000"
#define COLOR_normal    "000000"
#define COLOR_illegal   "FF0000"
#define COLOR_int       "2222AB"
#define COLOR_float     "2222AB"
#define COLOR_char      "2222AB"
#define COLOR_str       "0000FF"
#define COLOR_bool      "2222AB"
#define COLOR_null      "8877CC"
#define COLOR_comment   "33AA99"
#define COLOR_op        "AAAA00"
#define COLOR_seperator "767600"
#define COLOR_charKeyBoard  "888888"
#define COLOR_method    "00AAFF"
#define COLOR_id        "3388FF"
#define COLOR_newline   COLOR_background
#define COLOR_space     COLOR_background


#define FONT_normal     -1
#define FONT_bold       -2

#endif /* myscanner_h */

