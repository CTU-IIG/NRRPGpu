/*
	This file is part of the NRRPGpu program.

	NRRPGpu is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	NRRPGpu is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License

*/

#include "nrrp.cuh"

// Deep copy of array
void copyArray(unsigned int* destination, unsigned int* source, unsigned int length)
{
	unsigned int i;
	for(i = 0; i < length; i++)
		destination[i] = source[i];
}

// Get selected bit from array of bits
unsigned int getBitOfBitArray(unsigned int* array, int position)
{
	//int pos = (position/_dayCount)*32+(position%_dayCount);
	//return array[pos/32]&(1<<(pos%32));
	return array[position/_dayCount]&(1<<(position%_dayCount));
}

// Set selected bit in array of bits
void setBitOfBitArray(unsigned int* array, int position, int value)
{
	int pos = (position/_dayCount)*32+(position%_dayCount);
	if(value)
		array[pos/32] = array[pos/32]|(1<<(pos%32));
	else
		array[pos/32] = array[pos/32]&(INTMAX - (1<<(pos%32)));
}

// Conversation 2D index (row, column) to 1D index
unsigned int getIndex(unsigned int row, unsigned int column, unsigned int size)
{
	return row*size+column;
}

// Conversation 1D index to row
unsigned int getRow(unsigned int index, unsigned int size)
{
	return index/size;
}

// Conversation 1D index to column
unsigned int getColumn(unsigned int index, unsigned int size)
{
	return index%size;
}

unsigned int getNursePreference(unsigned int index, unsigned int shift)
{
	return _nursePreferences[index*4+shift];
}