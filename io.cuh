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

#ifndef IO_H_
#define IO_H_

void error(char* message);
int parseInput(int argc, char* argv[], int fileIndex);
int parseInputMaenhout(int argc, char* argv[], int fileIndex);
int parseInputMozPato(int argc, char* argv[], int fileIndex);
void printSolution();
void printStatistics(int instances);
void printRoster(unsigned int* currentRoster);
void printBitArray(unsigned int* bitArray);

#endif /* IO_H_ */
