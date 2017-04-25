#include <stdio.h>
#include <string.h>
#include "q.h"


#define true 1;
#define false 0;
extern int yylex();
extern char* yytext;
extern FILE* yyin;
FILE* outFile;


void print_text(char* color, int font){
    if(font == FONT_normal){
        fprintf(outFile,"<font color=\"%s\">", color);
        fprintf(outFile,"%s", yytext);
        fprintf(outFile,"%s", "</font>");
    }
    else if(font == FONT_bold){
        fprintf(outFile,"<strong><font color=\"%s\">", color);
        fprintf(outFile,"%s", yytext);
        fprintf(outFile,"%s", "</font></strong>");
    }
}
int is_legal(){
    
    return false;
}
void CHECK_legal_AND_print(){
    if(is_legal())
        print_text(COLOR_normal, FONT_normal);
    else
        print_text(COLOR_illegal,FONT_bold);
}


int main(int argc, char** argv){
    //argv[0] is this program name
    //argv[1] is param filename : <f>.java
    //this program's output is  : <f>.html
    yyin=fopen(argv[1], "r");
    char* _tmp = strtok(argv[1], ".java");
    char* _outFileName = strcat(_tmp, ".html");
    outFile = fopen(_outFileName, "w");
    fprintf(outFile, "%s", "<HTML><body>");
    

    int prev_state=-10000;
    int ENUM_token =yylex();
    while( 0< ENUM_token){
	if(prev_state == TYPE_type){
	    if(ENUM_token != TYPE_space){
		print_text(COLOR_space, FONT_normal);
       		ENUM_token =yylex();
		continue;
	    }
	    if(ENUM_token != TYPE_id){
       		print_text(COLOR_illegal,FONT_bold);
		prev_state =ENUM_token;
       		ENUM_token =yylex();
		continue;
	    }	
	}
        switch(ENUM_token){
            case TYPE_newline : fprintf(outFile,"%s", "<br>"); break;
            case TYPE_comment : print_text(COLOR_comment, FONT_normal); break;
            case TYPE_comment_newline : print_text(COLOR_comment, FONT_normal); fprintf(outFile,"%s", "<br>"); break;
            case TYPE_int : print_text(COLOR_int, FONT_normal); break;
            case TYPE_float : print_text(COLOR_float, FONT_normal); break;
            case TYPE_bool : print_text(COLOR_bool, FONT_normal); break;
            case TYPE_char : print_text(COLOR_char, FONT_normal); break;
            case TYPE_str : print_text(COLOR_str, FONT_normal); break;
            case TYPE_null : print_text(COLOR_null, FONT_normal); break;
            case TYPE_separator : print_text(COLOR_seperator, FONT_normal); break;
            case TYPE_charKeyBoard : print_text(COLOR_charKeyBoard, FONT_normal); break;
            case TYPE_id : print_text(COLOR_id, FONT_normal); break;
            case TYPE_method : print_text(COLOR_method, FONT_normal); break;
            case TYPE_space : print_text(COLOR_space, FONT_normal); break;
	    case TYPE_op : print_text(COLOR_op, FONT_normal); break;
            case TYPE_general : print_text(COLOR_int, FONT_normal); break;

            case TYPE_type : print_text(COLOR_int, FONT_normal); break;

            default: CHECK_legal_AND_print();
        }
	prev_state =ENUM_token;
        ENUM_token =yylex();
    }
    fprintf(outFile, "%s", "</body></HTML>");
    
    fclose(outFile);
    fclose(yyin);
    return 0;
}

