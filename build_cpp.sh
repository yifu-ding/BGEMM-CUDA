rm -rf build
mkdir build

/usr/local/bin/cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE=Debug -S./ -B./build  

echo "starting building executable file..."
/usr/local/bin/cmake  --build ./build --config Debug --target all -j 12 -- 

# ./build/test 256 256 256 1

