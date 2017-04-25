git clone https://github.com/helldongdae/compilerHW2
cd compilerHW2
chmod +x l.sh
lex q.l
gcc q.c lex.yy.c -lfl
./a.out code.java
firefox code.html
