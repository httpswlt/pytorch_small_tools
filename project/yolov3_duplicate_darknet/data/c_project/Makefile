image.so:
	gcc -fPIC -shared -o ./image.so ./*.c -I . `pkg-config --cflags --libs opencv`


clean:
	rm -f ./image.so