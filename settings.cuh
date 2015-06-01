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

#ifndef SETTINGS_H_
#define SETTINGS_H_

enum DataSetType {MOZPATO = 0, MAENHOUT = 1};

struct Settings{
	unsigned int changePenalization; // Penalisation for roster disruption

	unsigned int isFairness; // Does algorithm use fairness?
	unsigned int fairnessPenalization; // Penalization for fairness

	unsigned int isSevenDaySequenceConstrain; // Does algorithm use seven day sequence free constrain;

	unsigned int isLeftRightConstrainSoft; // Does algorithm manipulate with left and right hard constrain as with soft one
	unsigned int lrcsPenalization; // Penalization for left right soft constrain

	unsigned int isExtendedRoster; // Does algorithm use extended roster(few days before rostering period)

	unsigned int isDayShiftTypeMin; // Does algorithm use table of minimum count of particular shift type on particulal day

	unsigned int isShiftListOptimalization; // Does algorithm use optimalization algorithm for length of shiftList
	
	unsigned int isMinMaxHardConstrain; // Does algorithm use MinMax hard constrain
	unsigned int mmhcMinWorking; // Min number of working assignments
	unsigned int mmhcMaxWorking; // Max number of working assignments
	unsigned int mmhcMinConsecutive; // Min number of consecutive working assigments
	unsigned int mmhcMaxConsecutive; // Max number of consecutive working assigments
	unsigned int mmhcMinShiftType; // Min number of assignments per shift type
	unsigned int mmhcMaxShiftType; // Max number of assignments per shift type
	unsigned int mmhcMinShiftTypeConsecutive; // Min number of consecutive assignments per shift type
	unsigned int mmhcMaxShiftTypeConsecutive; // Max number of consecutive assignments per shift type

	unsigned int softInfeasiblePenalization; // Threshold of infeasibility(if infeasibility is handled as penalization)

	unsigned int isNursePreferences; // Does algorithm use nursePreferences policy

	unsigned int isFreezeShiftSupport; // Does algorithm use shift freezing(freezed shifts must be the same in new roster as in original one)
	
	DataSetType inputDataSetType; // Type of input dataset - 0=Moz&Pato, 1=Maenhout

	unsigned int isMulticommodityFlowConstrins; // Does algorithm use soft constrains from An Integer Multicommodity Flow Model Applied to the Rerostering of Nurse Schedules atricle
	unsigned int mfcNightPenalization; // Penalization of night shift on consecutive days
	unsigned int mfcPmPenalization; // Penalization of pm shift on more than three consecutive days
	unsigned int mfcFreePenalization; // Penalization for free shift on more than three consecutive days

	unsigned int isMPHeuristic; // Does algorithm use compatibility from Moz&Pato constructive heuristic
	unsigned int mphPenalization; // Penalization for compatibility 

};

extern Settings _settings;

void setMaenhout(); // Set algorithm for Maenhout dataset
void setMozPato(); // Set algorithm for Moz and Pato dataset
#endif /* SETTINGS_H_ */
