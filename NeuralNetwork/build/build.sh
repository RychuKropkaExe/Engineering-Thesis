rm -f CMakeCache.txt

cmake -G "Unix Makefiles" -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_COMPILER=g++ -DLOG_PRIO=1 -DDEBUG_MODE=0 ..

make
