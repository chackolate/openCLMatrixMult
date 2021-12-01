//hello, world!
__kernel void helloWorld(__global char* data){
	data[0] = 'H';
	data[1] = 'e';
	data[2] = 'l';
	data[3] = 'l';
	data[4] = 'o';
	data[5] = ',';
	data[6] = ' ';
	data[7] = 'W';
	data[8] = 'o';
	data[9] = 'r';
	data[10] = 'l';
	data[11] = 'd';
	data[12] = '!';
	data[13] = '\n';
}