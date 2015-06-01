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

#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

int isLeftHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster);
int isRightHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster);
int isHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* currentRoster);
int isLeftSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray);
int isRightSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray);
int isLeftCompatible(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray);
int isRightCompatible(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray);
int isNoOneSideSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray);
unsigned int CompatibilityPenalization(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster);
#endif /* CONSTRAINTS_H_ */
