%{
    #include <string.h>
    #include "q.h"
%}

%x COMMENTS
%x COMMENTS_LINE
%x CLASS_OP
%x STRING
%x VARorMETHOD
%x METHOD


TYPE                        default|case|abstract|assert|break|continue|do|else|extends|for|if|goto|instanceof|private|protected|public|while|super|volatile|strictfp|static|native|long|finally|final|const|try|catch|transient|throws|throw|this|synchronized|switch
OPERATORS                   [=<>!?~:\+\-\*\|\^&%/]|"=="|"<="|">="|"!="|"&&"|"||"|"<<"|">>"|"++"|"--"|"-="|"+="|"*="|"%="|"/="|"^="|"->"|"&="|"|="
SEPARATORS                  [(){}\[\];,\.]
TYPE2                       Class|interface|package|import|new|implements|throw|extends
TYPE3			    int|float|char|short|void|double|enum|return|boolean|byte


NULL                        null
BOOLEAN                     true|false
INT                         [+|-]?([1-9].[0-9]*)|[0]
FLOAT                       [+|-]?([1-9].[0-9]*)|[0]"."([0-9]*)
CHARACTER                   \'[^']\'
IDENTIFIERS                 [a-zA-Z][a-zA-Z$_0-9]*


%%
[\n]                        return TYPE_newline;
[ ]			    return TYPE_space;

"/*"                        {BEGIN(COMMENTS); return TYPE_comment;}
<COMMENTS>\n                return TYPE_comment_newline;
<COMMENTS>[^\"*/\"]*        return TYPE_comment;
<COMMENTS>"*/"              {BEGIN(INITIAL); return TYPE_comment;}

"//"                        {BEGIN(COMMENTS_LINE); return TYPE_comment;}
<COMMENTS_LINE>.*           return TYPE_comment;
<COMMENTS_LINE>\n           {BEGIN(INITIAL); return TYPE_comment;}

{NULL}                      return TYPE_null;
{BOOLEAN}                   return TYPE_bool;
{INT}                       return TYPE_int;
{FLOAT}                     return TYPE_float;
{CHARACTER}                 return TYPE_char;
\"                          {BEGIN(STRING); return TYPE_str;}
<STRING>[^"]*               return TYPE_str;
<STRING>\"                  {BEGIN(INITIAL); return TYPE_str;}

{TYPE}                      return TYPE_general;
{TYPE3}		 	    return TYPE_type;
{TYPE2}                     {BEGIN(CLASS_OP); return TYPE_type;}
<CLASS_OP>{IDENTIFIERS}     {BEGIN(INITIAL); return TYPE_general;}

{OPERATORS}                 return TYPE_op;
<*>SEPARATORS               return TYPE_separator;
.                           return TYPE_charKeyBoard;

{IDENTIFIERS}               return TYPE_id;




%%

