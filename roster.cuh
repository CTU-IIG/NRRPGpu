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

#ifndef ROASTER_H_
#define ROASTER_H_

void setRosterShift(unsigned int* roster, unsigned int index, int shift);
void setOrigRosterShift(unsigned int index, int shift);
void setExtendedRosterShift(unsigned int* roster, unsigned int index, int shift);
int getOrigRosterShift(unsigned int index);
int getRosterShift(unsigned int index, unsigned int* currentRoster);
int getExtendedRosterShift(unsigned int index, unsigned int* currentRoster);
int utilityFunction(unsigned int* currentRoster);
void shuffleShiftList(unsigned int* shiftList, int isLocal, float probability);


#endif /* ROASTER_H_ */
