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

#ifndef UTILITY_H_
#define UTILITY_H_

void copyArray(unsigned int* destination, unsigned int* source, unsigned int length);
unsigned int getBitOfBitArray(unsigned int* array, int position);
void setBitOfBitArray(unsigned int* array, int position, int value);
unsigned int getIndex(unsigned int row, unsigned int column, unsigned int size);
unsigned int getRow(unsigned int index, unsigned int size);
unsigned int getColumn(unsigned int index, unsigned int size);
unsigned int getNursePreference(unsigned int index, unsigned int shift);

#endif /* UTILITY_H_ */
