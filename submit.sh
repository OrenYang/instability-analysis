cp /Users/orenyang/FLASH/FLASH4.8/object/Noh_Example/flash4 /Users/orenyang/FLASH/FLASH4.8/SCRATCH/Noh/test_02
cp /Users/orenyang/FLASH/FLASH4.8/source/Simulation/SimulationMain/magnetoHD/unitTest/NohCylindricalRagelike/magnoh-analytic.txt /Users/orenyang/FLASH/FLASH4.8/SCRATCH/Noh/test_02

mpirun -np 4 ./flash4
