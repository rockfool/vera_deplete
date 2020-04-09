
"""
problem 2E FA 12 Pyrex TF=900K
moderator 600 K 
clad      600 K
fuel      900 K
moderator density 0.700 g/cc
power density 40.0 W/gU
"""

import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt
import json 

###############################################################################
#                      Simulation Input File Parameters
###############################################################################

# OpenMC simulation parameters
batches = 220
inactive = 20
particles = 100000

# Depletion simulation parameters
time_step = 1*24*60*60 # s
final_time = 5*24*60*60 # s
#time_steps = np.full(final_time // time_step, time_step)
#time_steps = np.array([0.25, 6.00, 6.25, 12.50, 25.00, 25.00, 25.00, # 100.00 EFPD 
#             25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, # 300.00 EFPD
#             25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, # 500.00 EFPD
#             62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, # 1000.0 EFPD
#             62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50  # 1500.0 EFPD
#             ])*24*60*60 
time_steps = np.array([0.25, 6.00, 6.25, 12.50, 25.00, 50.00,        # 100.00 EFPD 
             100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, # 900.00 EFPD
             100.0, 100.0, 100.0, 100.0, 100.0, 100.0                # 1500.0 EFPD
             ])*24*60*60 
#time_steps = np.array([0.25, 6.00, 6.25, 12.50, 25.00, 50.00,        # 100.00 EFPD 
#             50.0,  50.0, 50.0, 50.0, 100.0, 100.0, 500.0, 500.0     # 1500.00 EFPD
#             ])*24*60*60 
#chain_file = './chain_casl.xml'
power_den = 40.0 # W/cm, for 2D simulations only (use W for 3D) 40.0 W/gU 

###############################################################################
#                              Define materials
###############################################################################

# Instantiate some Materials and register the appropriate Nuclides
# Instantiate some Materials and register the appropriate Nuclides
fuel_21 = openmc.Material(material_id=1, name='UO2 fuel at 2.1% wt enrichment')
fuel_21.set_density('atom/b-cm', 6.86385E-02)
fuel_21.add_nuclide('U234',4.02487E-06 )
fuel_21.add_nuclide('U235',4.86484E-04 )
fuel_21.add_nuclide('U236',2.23756E-06 )
fuel_21.add_nuclide('U238',2.23868E-02 )
fuel_21.add_nuclide('O16' ,4.57590E-02 )
fuel_21.depletable = True


fuel_31 = openmc.Material(material_id=2, name='UO2 fuel at 3.1% wt enrichment')
fuel_31.set_density('atom/b-cm', 6.86463E-02)
fuel_31.add_nuclide('U234',6.11864E-06 )
fuel_31.add_nuclide('U235',7.18132E-04 )
fuel_31.add_nuclide('U236',3.29861E-06 )
fuel_31.add_nuclide('U238',2.21546E-02 )
fuel_31.add_nuclide('O16' ,4.57642E-02 )
fuel_31.depletable = True


fuel_36 = openmc.Material(material_id=3, name='UO2 fuel at 3.6% wt enrichment')
fuel_36.set_density('atom/b-cm', 6.86503E-02)
fuel_36.add_nuclide('U234',7.21203E-06 )
fuel_36.add_nuclide('U235',8.33952E-04 )
fuel_36.add_nuclide('U236',3.82913E-06 )
fuel_36.add_nuclide('U238',2.20384E-02 )
fuel_36.add_nuclide('O16' ,4.57669E-02 )
fuel_36.depletable = True


fuel_46 = openmc.Material(material_id=4, name='UO2 fuel at 4.6% wt enrichment')
fuel_46.set_density('atom/b-cm', 6.86582E-02)
fuel_46.add_nuclide('U234',9.39876E-06 )
fuel_46.add_nuclide('U235',1.06559E-03 )
fuel_46.add_nuclide('U236',4.89014E-06 )
fuel_46.add_nuclide('U238',2.18062E-02 )
fuel_46.add_nuclide('O16' ,4.57721E-02 )
fuel_46.depletable = True


helium = openmc.Material(material_id=5, name='Helium for gap')
helium.set_density('atom/b-cm',  2.68714E-5)
helium.add_nuclide('He4', 2.68714E-5)

zirc4 = openmc.Material(material_id=6, name='Zircaloy 4')
zirc4.set_density('atom/b-cm', 4.33818E-02)
zirc4.add_nuclide('Zr90',  2.18865E-02)
zirc4.add_nuclide('Zr91',  4.77292E-03)
zirc4.add_nuclide('Zr92',  7.29551E-03)
zirc4.add_nuclide('Zr94',  7.39335E-03)
zirc4.add_nuclide('Zr96',  1.19110E-03)
zirc4.add_nuclide('Sn112',  4.68066E-06)
zirc4.add_nuclide('Sn114',  3.18478E-06)
zirc4.add_nuclide('Sn115',  1.64064E-06)
zirc4.add_nuclide('Sn116',  7.01616E-05)
zirc4.add_nuclide('Sn117',  3.70592E-05)
zirc4.add_nuclide('Sn118',  1.16872E-04)
zirc4.add_nuclide('Sn119',  4.14504E-05)
zirc4.add_nuclide('Sn120',  1.57212E-04)
zirc4.add_nuclide('Sn122',  2.23417E-05)
zirc4.add_nuclide('Sn124',  2.79392E-05)
zirc4.add_nuclide('Fe54',  8.68307E-06)
zirc4.add_nuclide('Fe56',  1.36306E-04)
zirc4.add_nuclide('Fe57',  1.36306E-04)
zirc4.add_nuclide('Fe58',  4.18926E-07)
zirc4.add_nuclide('Cr50',  3.30121E-06)
zirc4.add_nuclide('Cr52',  6.36606E-05)
zirc4.add_nuclide('Cr53',  7.21860E-06)
zirc4.add_nuclide('Cr54',  1.79686E-06)
zirc4.add_nuclide('Hf174',  3.54138E-09)
zirc4.add_nuclide('Hf176',  1.16423E-07)
zirc4.add_nuclide('Hf177',  4.11686E-07)
zirc4.add_nuclide('Hf178',  6.03806E-07)
zirc4.add_nuclide('Hf179',  3.01460E-07)
zirc4.add_nuclide('Hf180',  7.76449E-07)

water_565 = openmc.Material(material_id=7, name='Borated water at 565 K with 1300 ppm')
water_565.set_density('atom/b-cm', 7.44874E-02)
water_565.add_nuclide('O16',2.48112E-02 )
water_565.add_nuclide('H1' ,4.96224E-02 )
water_565.add_nuclide('B10',1.07070E-05 )
water_565.add_nuclide('B11',4.30971E-05 )
water_565.add_s_alpha_beta('c_H_in_H2O')


water_600= openmc.Material(material_id=8, name='Borated water at 600 K with 1300 ppm')
water_600.set_density('atom/b-cm', 7.01765E-02)
water_600.add_nuclide('O16', 2.33753E-02)
water_600.add_nuclide('H1', 4.67505E-02)
water_600.add_nuclide('B10', 1.00874E-05)
water_600.add_nuclide('B11', 4.06030E-05)
water_600.add_s_alpha_beta('c_H_in_H2O')


ifba= openmc.Material(material_id=9, name='IFBA')
ifba.set_density('atom/b-cm', 6.19850E-02)
ifba.add_nuclide('B10',  2.16410E-02)
ifba.add_nuclide('B11',  1.96824E-02)
ifba.add_nuclide('Zr90', 1.06304E-02)
ifba.add_nuclide('Zr91', 2.31824E-03)
ifba.add_nuclide('Zr92', 3.54348E-03)
ifba.add_nuclide('Zr94', 3.59100E-03)
ifba.add_nuclide('Zr96', 5.78528E-04)


pyrex= openmc.Material(material_id=10, name='Pyrex')
pyrex.set_density('atom/b-cm', 7.13737E-02)
pyrex.add_nuclide('B10', 9.63266E-04) 
pyrex.add_nuclide('B11', 3.90172E-03)
pyrex.add_nuclide('O16', 4.67761E-02)
pyrex.add_nuclide('Si28', 1.81980E-02)
pyrex.add_nuclide('Si29', 9.24474E-04)
pyrex.add_nuclide('Si30', 6.10133E-04)


gad= openmc.Material(material_id=11, name='gad with 5% Gd2O3 95% UO2, 1.8 % U-235')
gad.set_density('atom/b-cm', 6.84756E-02 )
gad.add_nuclide('U234', 3.18096E-06)
gad.add_nuclide('U235', 3.90500E-04)
gad.add_nuclide('U236', 1.79300E-06)
gad.add_nuclide('U238', 2.10299E-02)
gad.add_nuclide('Gd152', 3.35960E-06)
gad.add_nuclide('Gd154', 3.66190E-05)
gad.add_nuclide('Gd155', 2.48606E-04)
gad.add_nuclide('Gd156', 3.43849E-04)
gad.add_nuclide('Gd157', 2.62884E-04)
gad.add_nuclide('Gd158', 4.17255E-04)
gad.add_nuclide('Gd160', 3.67198E-04)
gad.add_nuclide('O16'  , 4.53705E-02)
gad.depletable = True


ss304= openmc.Material(material_id=12, name='SS304')
ss304.set_density('atom/b-cm', 8.82490E-02)
ss304.add_element('C'  ,  3.20895E-04)
ss304.add_nuclide('Si28', 1.58197E-03)
ss304.add_nuclide('Si29', 8.03653E-05)
ss304.add_nuclide('Si30', 5.30394E-05)
ss304.add_nuclide('P31',  6.99938E-05)
ss304.add_nuclide('Cr50', 7.64915E-04)
ss304.add_nuclide('Cr52', 1.47506E-02)
ss304.add_nuclide('Cr53', 1.67260E-03)
ss304.add_nuclide('Cr54', 4.16346E-04)
ss304.add_nuclide('Mn55', 1.75387E-03)
ss304.add_nuclide('Fe54', 3.44776E-03)
ss304.add_nuclide('Fe56', 5.41225E-02)
ss304.add_nuclide('Fe57', 1.24992E-03)
ss304.add_nuclide('Fe58', 1.66342E-04)
ss304.add_nuclide('Ni58', 5.30854E-03)
ss304.add_nuclide('Ni60', 2.04484E-03)
ss304.add_nuclide('Ni61', 8.88879E-05)
ss304.add_nuclide('Ni62', 2.83413E-04)
ss304.add_nuclide('Ni64', 7.21770E-05)


agincd= openmc.Material(material_id=13, name='Ag-In-Cd')
agincd.set_density('atom/b-cm', 5.63131E-02)
agincd.add_nuclide('Ag107', 2.36159E-02)
agincd.add_nuclide('Ag109', 2.19403E-02)
agincd.add_nuclide('Cd106', 3.41523E-05)
agincd.add_nuclide('Cd108', 2.43165E-05)
agincd.add_nuclide('Cd110', 3.41250E-04)
agincd.add_nuclide('Cd111', 3.49720E-04)
agincd.add_nuclide('Cd112', 6.59276E-04)
agincd.add_nuclide('Cd113', 3.33873E-04)
agincd.add_nuclide('Cd114', 7.84957E-04)
agincd.add_nuclide('Cd116', 2.04641E-04)
agincd.add_nuclide('In113', 3.44262E-04)
agincd.add_nuclide('In115', 7.68050E-03)

b4c= openmc.Material(material_id=14, name='B4C')
b4c.set_density('atom/b-cm', 9.59100E-02 )
b4c.add_nuclide('B10',1.52689E-02)
b4c.add_nuclide('B11',6.14591E-02)
b4c.add_element('C'  ,1.91820E-02)

waba= openmc.Material(material_id=15, name='WABA B4C-Al2O3')
waba.set_density('atom/b-cm', 1.16453E-01)
waba.add_nuclide('B10', 2.98553E-03)
waba.add_nuclide('B11', 1.21192E-02)
waba.add_element('C'  , 3.77001E-03)
waba.add_nuclide('O16', 5.85563E-02)
waba.add_nuclide('Al27',3.90223E-02)

materials_file = openmc.Materials([fuel_21, fuel_31, fuel_36, fuel_46, helium, water_565, water_600, pyrex, zirc4, gad, ifba, ss304, agincd, b4c, waba])
materials_file.export_to_xml()

###############################################################################
#                             Create geometry
###############################################################################

# Instantiate zCylinder surfaces
fuel_or = openmc.ZCylinder(surface_id=1, x0=0, y0=0, r=0.4096, name='Fuel OR')
clad_ir = openmc.ZCylinder(surface_id=2, x0=0, y0=0, r=0.4180, name='Clad IR')
clad_or = openmc.ZCylinder(surface_id=3, x0=0, y0=0, r=0.4750, name='Clad OR')
gt_ir   = openmc.ZCylinder(surface_id=4, x0=0, y0=0, r=0.5610, name='Guide Tube IR')
gt_or   = openmc.ZCylinder(surface_id=5, x0=0, y0=0, r=0.6020, name='Guide Tube OR')
it_ir   = openmc.ZCylinder(surface_id=6, x0=0, y0=0, r=0.5590, name='Instr Tube IR')
it_or   = openmc.ZCylinder(surface_id=7, x0=0, y0=0, r=0.6050, name='Instr Tube OR')
th_ir   = openmc.ZCylinder(surface_id=8, x0=0, y0=0, r=0.2580, name='Thimble IR')
th_or   = openmc.ZCylinder(surface_id=9, x0=0, y0=0, r=0.3820, name='Thimble OR')
#
aic_ir       = openmc.ZCylinder(surface_id=11, x0=0, y0=0, r=0.3820, name='AIC Radius')
aic_clad_ir  = openmc.ZCylinder(surface_id=12, x0=0, y0=0, r=0.3860, name='AIC Clad IR')
aic_clad_or  = openmc.ZCylinder(surface_id=13, x0=0, y0=0, r=0.4840, name='AIC Clad OR')
b4c_ir       = openmc.ZCylinder(surface_id=14, x0=0, y0=0, r=0.3730, name='B4C Radius')
#
ifba_or      = openmc.ZCylinder(surface_id=21, x0=0, y0=0, r=0.4106, name='IFBA OR')
#
waba_clad1_ir = openmc.ZCylinder(surface_id=31, x0=0, y0=0, r=0.2860, name='WABA Clad 1 IR')
waba_clad1_or = openmc.ZCylinder(surface_id=32, x0=0, y0=0, r=0.3390, name='WABA Clad 1 OR')
waba_ir       = openmc.ZCylinder(surface_id=33, x0=0, y0=0, r=0.3530, name='WABA IR')
waba_or       = openmc.ZCylinder(surface_id=34, x0=0, y0=0, r=0.4040, name='WABA OR')
waba_clad2_ir = openmc.ZCylinder(surface_id=35, x0=0, y0=0, r=0.4180, name='WABA Clad 2 IR')
waba_clad2_or = openmc.ZCylinder(surface_id=36, x0=0, y0=0, r=0.4840, name='WABA Clad 2 OR')
#
pyrex_clad1_ir      = openmc.ZCylinder(surface_id=41, x0=0, y0=0, r=0.2140, name='PYREX Clad 1 IR')
pyrex_clad1_or      = openmc.ZCylinder(surface_id=42, x0=0, y0=0, r=0.2310, name='PYREX Clad 1 OR')
pyrex_ir            = openmc.ZCylinder(surface_id=43, x0=0, y0=0, r=0.2410, name='PYREX IR')
pyrex_or            = openmc.ZCylinder(surface_id=44, x0=0, y0=0, r=0.4270, name='PYREX OR')
pyrex_clad2_ir      = openmc.ZCylinder(surface_id=45, x0=0, y0=0, r=0.4370, name='PYREX Clad 2 IR')
pyrex_clad2_or      = openmc.ZCylinder(surface_id=46, x0=0, y0=0, r=0.4840, name='PYREX Clad 2 OR')
#
pin_left   = openmc.XPlane(surface_id=101,  x0=-0.6300, name='left')
pin_right  = openmc.XPlane(surface_id=102,  x0= 0.6300, name='right')
pin_back   = openmc.YPlane(surface_id=103,  y0=-0.6300, name='back')
pin_front  = openmc.YPlane(surface_id=104,  y0= 0.6300, name='front')
pin_bottom = openmc.ZPlane(surface_id=105,  z0=-200.0, name='bottom')
pin_top    = openmc.ZPlane(surface_id=106,  z0= 200.0, name='top')

pin_left.boundary_type   = 'vacuum'
pin_right.boundary_type  = 'vacuum'
pin_back.boundary_type   = 'vacuum'
pin_front.boundary_type  = 'vacuum'
pin_bottom.boundary_type = 'vacuum'
pin_top.boundary_type    = 'vacuum'
#
assembly_left   = openmc.XPlane(surface_id=201,  x0=-10.75, name='left')
assembly_right  = openmc.XPlane(surface_id=202,  x0= 10.75, name='right')
assembly_back   = openmc.YPlane(surface_id=203,  y0=-10.75, name='back')
assembly_front  = openmc.YPlane(surface_id=204,  y0= 10.75, name='front')
assembly_bottom = openmc.ZPlane(surface_id=205,  z0=-200.0, name='bottom')
assembly_top    = openmc.ZPlane(surface_id=206,  z0= 200.0, name='top')
#
lattice_left    = openmc.XPlane(surface_id=211,  x0=-10.71, name='lat left')
lattice_right   = openmc.XPlane(surface_id=212,  x0= 10.71, name='lat right')
lattice_back    = openmc.YPlane(surface_id=213,  y0=-10.71, name='lat back')
lattice_front   = openmc.YPlane(surface_id=214,  y0= 10.71, name='lat front')
lattice_bottom  = openmc.ZPlane(surface_id=215,  z0=-200.0, name='lat bottom')
lattice_top     = openmc.ZPlane(surface_id=216,  z0= 200.0, name='lat top')
#
assembly_left.boundary_type   = 'reflective'
assembly_right.boundary_type  = 'reflective'
assembly_back.boundary_type   = 'reflective'
assembly_front.boundary_type  = 'reflective'
assembly_bottom.boundary_type = 'reflective'
assembly_top.boundary_type    = 'reflective'

# Instantiate cells
full_model  = openmc.Cell(cell_id=1, name='full core') 
#
lat_inner   = openmc.Cell(cell_id=2, name='assembly inside')
lat_outer   = openmc.Cell(cell_id=3, name='assembly outer')
#
# fuel rod cell with 3.6 w/o
fuel_pellet = openmc.Cell(cell_id=11, name='cell 11')   
fuel_gap    = openmc.Cell(cell_id=12, name='cell 12')   
fuel_clad   = openmc.Cell(cell_id=13, name='cell 13')   
fuel_water  = openmc.Cell(cell_id=14, name='cell 14')   
# fuel rod cell with 3.1 w/o
fl36_pellet = openmc.Cell(cell_id=15, name='cell 15')   
fl36_gap    = openmc.Cell(cell_id=16, name='cell 16')   
fl36_clad   = openmc.Cell(cell_id=17, name='cell 17')   
fl36_water  = openmc.Cell(cell_id=18, name='cell 18')   
# guide tube cell
gt_inner    = openmc.Cell(cell_id=21, name='cell 21')   
gt_mat      = openmc.Cell(cell_id=22, name='cell 22')   
gt_outer    = openmc.Cell(cell_id=23, name='cell 23')   
# instrument tube cell
it_th_inner = openmc.Cell(cell_id=31, name='cell 31')   
it_th_mat   = openmc.Cell(cell_id=32, name='cell 32')   
it_inner    = openmc.Cell(cell_id=33, name='cell 33')   
it_mat      = openmc.Cell(cell_id=34, name='cell 34')   
it_outer    = openmc.Cell(cell_id=35, name='cell 35')   
# gadolinia rod cell 
gad_pellet = openmc.Cell(cell_id=41, name='cell 41')    
gad_gap    = openmc.Cell(cell_id=42, name='cell 42')    
gad_clad   = openmc.Cell(cell_id=43, name='cell 43')    
gad_water  = openmc.Cell(cell_id=44, name='cell 44')    
# control rod cell Ag-In-Cd (AIC) 
aic_mat   =  openmc.Cell(cell_id=51, name='cell 51')    
aic_gap   =  openmc.Cell(cell_id=52, name='cell 52')    
aic_clad  =  openmc.Cell(cell_id=53, name='cell 53')      
aic_gt_in =  openmc.Cell(cell_id=54, name='cell 54') 
aic_gt_mat=  openmc.Cell(cell_id=55, name='cell 55') 
aic_gt_out=  openmc.Cell(cell_id=56, name='cell 56') 
# control rod cell B4C 
b4c_mat   =  openmc.Cell(cell_id=61, name='cell 61')    
b4c_gap   =  openmc.Cell(cell_id=62, name='cell 62')    
b4c_clad  =  openmc.Cell(cell_id=63, name='cell 63')      
b4c_gt_in =  openmc.Cell(cell_id=64, name='cell 64') 
b4c_gt_mat=  openmc.Cell(cell_id=65, name='cell 65') 
b4c_gt_out=  openmc.Cell(cell_id=66, name='cell 66') 
# pyrex rod cell 
pyr_inner    =  openmc.Cell(cell_id=71, name='cell 71') 
pyr_clad_1   =  openmc.Cell(cell_id=72, name='cell 72') 
pyr_gap_1    =  openmc.Cell(cell_id=73, name='cell 73') 
pyr_mat      =  openmc.Cell(cell_id=74, name='cell 74') 
pyr_gap_2    =  openmc.Cell(cell_id=75, name='cell 75') 
pyr_clad_2   =  openmc.Cell(cell_id=76, name='cell 76') 
pyr_gt_in    =  openmc.Cell(cell_id=77, name='cell 77') 
pyr_gt_mat   =  openmc.Cell(cell_id=78, name='cell 78') 
pyr_gt_out   =  openmc.Cell(cell_id=79, name='cell 79') 
# ifba rod cell 
ifba_pellet =  openmc.Cell(cell_id=81, name='cell 81')  
ifba_mat    =  openmc.Cell(cell_id=82, name='cell 82')  
ifba_gap    =  openmc.Cell(cell_id=83, name='cell 83')  
ifba_clad   =  openmc.Cell(cell_id=84, name='cell 84')  
ifba_outer  =  openmc.Cell(cell_id=85, name='cell 85')  
# waba rod cell 
waba_inner    =  openmc.Cell(cell_id=91, name='cell 91') 
waba_clad_1   =  openmc.Cell(cell_id=92, name='cell 92') 
waba_gap_1    =  openmc.Cell(cell_id=93, name='cell 93') 
waba_mat      =  openmc.Cell(cell_id=94, name='cell 94') 
waba_gap_2    =  openmc.Cell(cell_id=95, name='cell 95') 
waba_clad_2   =  openmc.Cell(cell_id=96, name='cell 96') 
waba_gt_in    =  openmc.Cell(cell_id=97, name='cell 97') 
waba_gt_mat   =  openmc.Cell(cell_id=98, name='cell 98') 
waba_gt_out   =  openmc.Cell(cell_id=99, name='cell 99') 

#full_model  . temperature=565.0
# fuel rod c                   
fuel_pellet . temperature=900.0
fuel_gap    . temperature=600.0
fuel_clad   . temperature=600.0
fuel_water  . temperature=600.0
# fuel rod c                   
fl36_pellet . temperature=900.0
fl36_gap    . temperature=600.0
fl36_clad   . temperature=600.0
fl36_water  . temperature=600.0
# guide tube                   
gt_inner    . temperature=600.0
gt_mat      . temperature=600.0 
gt_outer    . temperature=600.0
# instrument                   
it_th_inner . temperature=600.0
it_th_mat   . temperature=600.0
it_inner    . temperature=600.0
it_mat      . temperature=600.0
it_outer    . temperature=600.0
# gadolinia                     
gad_pellet .  temperature=900.0
gad_gap    .  temperature=600.0
gad_clad   .  temperature=600.0
gad_water  .  temperature=600.0
# control rod                    
aic_mat   .   temperature=600.0
aic_gap   .   temperature=600.0
aic_clad  .   temperature=600.0
aic_gt_in .   temperature=600.0
aic_gt_mat.   temperature=600.0
aic_gt_out.   temperature=600.0
# control rod                   
b4c_mat   .   temperature=600.0
b4c_gap   .   temperature=600.0
b4c_clad  .   temperature=600.0
b4c_gt_in .   temperature=600.0
b4c_gt_mat.   temperature=600.0
b4c_gt_out.   temperature=600.0
# pyrex rod                    
pyr_inner  .  temperature=600.0
pyr_clad_1 .  temperature=600.0
pyr_gap_1  .  temperature=600.0
pyr_mat    .  temperature=600.0
pyr_gap_2  .  temperature=600.0
pyr_clad_2 .  temperature=600.0
pyr_gt_in  .  temperature=600.0
pyr_gt_mat .  temperature=600.0
pyr_gt_out .  temperature=600.0
# ifba rod c                   
ifba_pellet.  temperature=900.0
ifba_mat   .  temperature=600.0
ifba_gap   .  temperature=600.0
ifba_clad  .  temperature=600.0
ifba_outer .  temperature=600.0
# waba 
waba_inner .  temperature=600.0
waba_clad_1.  temperature=600.0
waba_gap_1 .  temperature=600.0
waba_mat   .  temperature=600.0
waba_gap_2 .  temperature=600.0
waba_clad_2.  temperature=600.0
waba_gt_in .  temperature=600.0
waba_gt_mat.  temperature=600.0
waba_gt_out.  temperature=600.0

# Use surface half-spaces to define regions
full_model.region  = +assembly_left & -assembly_right & +assembly_back & -assembly_front
# lattice inside 
lat_inner.region   = +lattice_left  & -lattice_right  & +lattice_back  & -lattice_front
# lattice outside
lat_outer.region   = -lattice_left  | +lattice_right  | -lattice_back  | +lattice_front 
# fuel rod definition for 3.1 w/o
fuel_pellet.region = -fuel_or
fuel_gap.region    = +fuel_or & -clad_ir
fuel_clad.region   = +clad_ir & -clad_or
fuel_water.region  = +clad_or 
# fuel rod definiton for 3.6 w/o
fl36_pellet.region = -fuel_or
fl36_gap.region    = +fuel_or & -clad_ir
fl36_clad.region   = +clad_ir & -clad_or
fl36_water.region  = +clad_or 
# guide tube #
gt_inner.region    = -gt_ir 
gt_mat.region      = +gt_ir & -gt_or
gt_outer.region    = +gt_or
# instrument 
it_th_inner.region = -th_ir
it_th_mat.region   = +th_ir & -th_or
it_inner.region    = +th_or & it_ir
it_mat.region      = +it_ir & -it_or
it_outer.region    = +it_or
# gadolinia rod 
gad_pellet.region = -fuel_or 
gad_gap.region    = +fuel_or & -clad_ir
gad_clad.region   = +clad_ir & -clad_or
gad_water.region  = +clad_or
# control rod
aic_mat.region   = -aic_ir 
aic_gap.region   = +aic_ir & -aic_clad_ir 
aic_clad.region  = +aic_clad_ir & -aic_clad_or  
aic_gt_in .region= +aic_clad_or & -gt_ir
aic_gt_mat.region= +gt_ir  & -gt_or
aic_gt_out.region= +gt_or
# control rod
b4c_mat.region   = -b4c_ir
b4c_gap.region   = +b4c_ir & -aic_clad_ir
b4c_clad.region  = +aic_clad_ir & -aic_clad_or  
b4c_gt_in .region= +aic_clad_or & -gt_ir
b4c_gt_mat.region= +gt_ir  & -gt_or
b4c_gt_out.region= +gt_or
# pyrex rod c
pyr_inner.region  = -pyrex_clad1_ir     
pyr_clad_1.region = +pyrex_clad1_ir & -pyrex_clad1_or
pyr_gap_1.region  = +pyrex_clad1_or & -pyrex_ir    
pyr_mat.region    = +pyrex_ir & -pyrex_or
pyr_gap_2.region  = +pyrex_or & -pyrex_clad2_ir    
pyr_clad_2.region = +pyrex_clad2_ir & -pyrex_clad2_or   
pyr_gt_in.region  = +pyrex_clad2_or & -gt_ir
pyr_gt_mat.region = +gt_ir & -gt_or
pyr_gt_out.region = +gt_or
# ifba rod ce
ifba_pellet.region = -fuel_or
ifba_mat.region    = +fuel_or & -ifba_or 
ifba_gap.region    = +ifba_or & -clad_ir
ifba_clad.region   = +clad_ir & -clad_or
ifba_outer.region  = +clad_or
# waba rod ce
waba_inner.region  = -waba_clad1_ir
waba_clad_1.region = +waba_clad1_ir & -waba_clad1_or
waba_gap_1.region  = +waba_clad1_or & -waba_ir
waba_mat.region    = +waba_ir & -waba_or     
waba_gap_2.region  = +waba_or & -waba_clad2_ir
waba_clad_2.region = +waba_clad2_ir & -waba_clad2_or
aic_gt_in .region= +waba_clad2_or & -gt_ir
aic_gt_mat.region= +gt_ir  & -gt_or
aic_gt_out.region= +gt_or

# fill in the materials in each cell 
lat_outer.fill   = water_600
# fuel rod 3.1 w/o 
fuel_pellet.fill = fuel_31
fuel_gap.fill    = helium 
fuel_clad.fill   = zirc4
fuel_water.fill  = water_600
# fuel rod 3.6 w/o 
fl36_pellet.fill = fuel_36
fl36_gap.fill    = helium 
fl36_clad.fill   = zirc4
fl36_water.fill  = water_600
# guide tube #
gt_inner.fill    = water_600
gt_mat.fill      = zirc4
gt_outer.fill    = water_600
# instrument 
it_th_inner.fill = helium
it_th_mat.fill   = ss304
it_inner.fill    = water_600
it_mat.fill      = zirc4
it_outer.fill    = water_600
# gadolinia rod 
gad_pellet.fill = gad
gad_gap.fill    = helium
gad_clad.fill   = zirc4
gad_water.fill  = water_600
# control rod
aic_mat.fill   = agincd
aic_gap.fill   = helium
aic_clad.fill  = ss304 
aic_gt_in.fill = water_600
aic_gt_mat.fill= zirc4
aic_gt_out.fill= water_600
# control rod
b4c_mat.fill   = b4c 
b4c_gap.fill   = helium 
b4c_clad.fill  = ss304 
b4c_gt_in.fill = water_600
b4c_gt_mat.fill= zirc4
b4c_gt_out.fill= water_600
# pyrex rod c
pyr_inner.fill  = helium #water_600
pyr_clad_1.fill = ss304  
pyr_gap_1.fill  = helium
pyr_mat.fill    = pyrex      
pyr_gap_2.fill  = helium
pyr_clad_2.fill = ss304   
pyr_gt_in.fill  = water_600 
pyr_gt_mat.fill = zirc4
pyr_gt_out.fill = water_600   
# ifba rod ce
ifba_pellet.fill = fuel_31
ifba_mat.fill    = ifba
ifba_gap.fill    = helium
ifba_clad.fill   = zirc4
ifba_outer.fill  = water_600
# waba rod ce
waba_inner.fill  = water_600  
waba_clad_1.fill = zirc4 
waba_gap_1.fill  = helium  
waba_mat.fill    = waba     
waba_gap_2.fill  = helium  
waba_clad_2.fill = zirc4 
waba_gt_in.fill  = water_600
waba_gt_mat.fill = zirc4
waba_gt_out.fill = water_600

# Instantiate universe
root = openmc.Universe(universe_id=0, name='root universe')
# assembly universe 
u_assembly_2a = openmc.Universe(universe_id=21,name='infinity assembly universe 2a')
u_assembly_2e = openmc.Universe(universe_id=22,name='infinity assembly universe 2e')
u_assembly_2f = openmc.Universe(universe_id=23,name='infinity assembly universe 2f')
u_assembly_2g = openmc.Universe(universe_id=24,name='infinity assembly universe 2g')
u_assembly_2h = openmc.Universe(universe_id=25,name='infinity assembly universe 2h')
u_assembly_2i = openmc.Universe(universe_id=26,name='infinity assembly universe 2i')
u_assembly_2j = openmc.Universe(universe_id=27,name='infinity assembly universe 2j')
u_assembly_2k = openmc.Universe(universe_id=28,name='infinity assembly universe 2k')
u_assembly_2l = openmc.Universe(universe_id=29,name='infinity assembly universe 2l')
u_assembly_2m = openmc.Universe(universe_id=30,name='infinity assembly universe 2m')
u_assembly_2n = openmc.Universe(universe_id=31,name='infinity assembly universe 2n')
u_assembly_2o = openmc.Universe(universe_id=32,name='infinity assembly universe 2o')
u_assembly_2p = openmc.Universe(universe_id=33,name='infinity assembly universe 2p')
# other universes
u_fuel = openmc.Universe(universe_id=101)
u_fl36 = openmc.Universe(universe_id=102)
u_gad  = openmc.Universe(universe_id=103)
u_gt   = openmc.Universe(universe_id=104)
u_it   = openmc.Universe(universe_id=105)
u_aic  = openmc.Universe(universe_id=106)
u_b4c  = openmc.Universe(universe_id=107)
u_ifba = openmc.Universe(universe_id=108)
u_waba = openmc.Universe(universe_id=109)
u_pyr  = openmc.Universe(universe_id=200)
# fill unvierses with cells
u_fuel.add_cells([fuel_pellet, fuel_gap, fuel_clad, fuel_water])
u_fl36.add_cells([fl36_pellet, fl36_gap, fl36_clad, fl36_water]) 
u_gad .add_cells([ gad_pellet,  gad_gap,  gad_clad,  gad_water]) 
u_gt  .add_cells([   gt_inner,   gt_mat,  gt_outer]) 
u_it  .add_cells([it_th_inner,it_th_mat,  it_inner,  it_mat, it_outer]) 
u_aic .add_cells([    aic_mat,  aic_gap,  aic_clad,  aic_gt_in, aic_gt_mat, aic_gt_out]) 
u_b4c .add_cells([    b4c_mat,  b4c_gap,  b4c_clad,  b4c_gt_in, b4c_gt_mat, b4c_gt_out]) 
u_ifba.add_cells([ifba_pellet, ifba_mat,  ifba_gap,  ifba_clad, ifba_outer]) 
u_waba.add_cells([ waba_inner, waba_clad_1, waba_gap_1, waba_mat, waba_gap_2, waba_clad_2, waba_gt_in, waba_gt_mat, waba_gt_out]) 
u_pyr .add_cells([  pyr_inner,  pyr_clad_1,  pyr_gap_1,  pyr_mat,  pyr_gap_2,  pyr_clad_2,  pyr_gt_in, pyr_gt_mat, pyr_gt_out]) 
# fill assembly with cells
u_assembly_2a.add_cells([lat_inner, lat_outer])

# Register cells with Universe
root.add_cells([full_model])
# lattice of whole core
lattice_core = openmc.RectLattice(lattice_id=1001)
lattice_core.lower_left = [-10.75, -10.75]
lattice_core.pitch      = [21.50, 21.50]
lattice_core.universes  =[[u_assembly_2a]] # 1-by-1 whole core


# Instantiate a Lattice
lattice_2a = openmc.RectLattice(lattice_id=1)
lattice_2a.lower_left = [-10.71, -10.71]
lattice_2a.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2a.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2e = openmc.RectLattice(lattice_id=2)
lattice_2e.lower_left = [-10.71, -10.71]
lattice_2e.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2e.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2f = openmc.RectLattice(lattice_id=3)
lattice_2f.lower_left = [-10.71, -10.71]
lattice_2f.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2f.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2g = openmc.RectLattice(lattice_id=4)
lattice_2g.lower_left = [-10.71, -10.71]
lattice_2g.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2g.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_aic , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 



lattice_2h = openmc.RectLattice(lattice_id=5)
lattice_2h.lower_left = [-10.71, -10.71]
lattice_2h.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2h.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_b4c , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 



lattice_2i = openmc.RectLattice(lattice_id=6)
lattice_2i.lower_left = [-10.71, -10.71]
lattice_2i.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2i.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_it  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2j = openmc.RectLattice(lattice_id=7)
lattice_2j.lower_left = [-10.71, -10.71]
lattice_2j.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2j.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_it  , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_pyr , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2k = openmc.RectLattice(lattice_id=8)
lattice_2k.lower_left = [-10.71, -10.71]
lattice_2k.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2k.universes  = [ [u_fuel, u_fuel, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fuel], # 2
                          [u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36, u_fl36, u_fuel], # 3
                          [u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36], # 4 
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 5
                          [u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36], # 6
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 7
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 8
                          [u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fuel, u_gt  , u_fuel, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36], # 9
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 10
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 11
                          [u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36], # 12
                          [u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36], # 13
                          [u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36], # 14
                          [u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_pyr , u_fl36, u_fl36, u_fl36, u_fl36, u_fuel], # 15
                          [u_fuel, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fl36, u_fuel, u_fuel, u_fuel] ] # 17 



lattice_2l = openmc.RectLattice(lattice_id=9)
lattice_2l.lower_left = [-10.71, -10.71]
lattice_2l.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2l.universes  = [ [u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_fuel, u_ifba, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_ifba, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_ifba, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_fuel], # 5
                          [u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 8
                          [u_fuel, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_fuel], # 9
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_fuel], # 12
                          [u_fuel, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_ifba, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_ifba, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_gt  , u_fuel, u_ifba, u_gt  , u_ifba, u_fuel, u_gt  , u_ifba, u_fuel, u_ifba, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba] ] # 17 



lattice_2m = openmc.RectLattice(lattice_id=10)
lattice_2m.lower_left = [-10.71, -10.71]
lattice_2m.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2m.universes  = [ [u_ifba, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_ifba], # 1
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 2
                          [u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_fuel, u_ifba, u_fuel], # 3
                          [u_fuel, u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_ifba, u_gt  , u_ifba, u_fuel, u_fuel], # 4 
                          [u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba], # 5
                          [u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_fuel], # 6
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 7
                          [u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba], # 8
                          [u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_fuel], # 9
                          [u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba], # 10
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 11
                          [u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_fuel], # 12
                          [u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba], # 13
                          [u_fuel, u_fuel, u_ifba, u_gt  , u_ifba, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_ifba, u_gt  , u_ifba, u_fuel, u_fuel], # 14
                          [u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_fuel, u_ifba, u_fuel], # 15
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 16
                          [u_ifba, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_ifba] ] # 17 


lattice_2n = openmc.RectLattice(lattice_id=11)
lattice_2n.lower_left = [-10.71, -10.71]
lattice_2n.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2n.universes  = [ [u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_fuel, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_fuel, u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_ifba, u_waba, u_ifba, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_ifba, u_waba, u_ifba, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_fuel], # 5
                          [u_fuel, u_ifba, u_waba, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_waba, u_ifba, u_fuel], # 6
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 8
                          [u_fuel, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_fuel], # 9
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel], # 11
                          [u_fuel, u_ifba, u_waba, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_gt  , u_ifba, u_ifba, u_waba, u_ifba, u_fuel], # 12
                          [u_fuel, u_fuel, u_ifba, u_ifba, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_ifba, u_ifba, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_ifba, u_waba, u_ifba, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_ifba, u_waba, u_ifba, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_fuel, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_waba, u_ifba, u_ifba, u_fuel, u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_ifba, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_ifba] ] # 17 


lattice_2o = openmc.RectLattice(lattice_id=12)
lattice_2o.lower_left = [-10.71, -10.71]
lattice_2o.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2o.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gad , u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gad , u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 


lattice_2p = openmc.RectLattice(lattice_id=13)
lattice_2p.lower_left = [-10.71, -10.71]
lattice_2p.pitch      = [1.2600, 1.2600]
                          # 1        2       3      4       5        6       7       8       9       10     11       12      13     14      15      16      17
lattice_2p.universes  = [ [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 1
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 2
                          [u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gad , u_fuel, u_fuel], # 3
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 4 
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel], # 5
                          [u_fuel, u_gad , u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_gad , u_fuel], # 6
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 7
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 8
                          [u_fuel, u_fuel, u_gt  , u_gad , u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_gad , u_gt  , u_fuel, u_fuel], # 9
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 10
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 11
                          [u_fuel, u_gad , u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_gad , u_fuel], # 12
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel], # 13
                          [u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_fuel], # 14
                          [u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gt  , u_fuel, u_fuel, u_gad , u_fuel, u_fuel], # 15
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_gad , u_fuel, u_fuel, u_fuel, u_fuel, u_fuel], # 16
                          [u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel, u_fuel] ] # 17 

# Fill cell with the lattice
lat_inner.fill  = lattice_2e
full_model.fill = lattice_core


# Instantiate a Geometry, register the root Universe
geometry = openmc.Geometry(root)
geometry.export_to_xml()
###############################################################################
#                     Transport calculation settings
###############################################################################

# Instantiate a Settings object, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
#  temp_method = {}
#  temp_method['method'] = 'nearest'
#  temp_method['range'] = 500.0
#  temp_method['multipole'] = True
#  settings_file.temperature = temp_method
#
settings_file.temperature={
		 'method'    :'nearest',
		 'tolerance' : 100.0,
		 'multipole' : False
 }
#

settings_file.resonance_scattering = {
                 'enable'    : False,
                 'method'    : 'dbrc'
}
# Create an initial uniform spatial source distribution over fissionable zones
bounds = [-10.75, -10.75, -1, 10.75, 10.75, 1]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

entropy_mesh = openmc.RegularMesh()
entropy_mesh.lower_left  = [-10.71, -10.71, -1.e50]
entropy_mesh.upper_right = [ 10.71,  10.71,  1.e50]
entropy_mesh.dimension = [10, 10, 1]
settings_file.entropy_mesh = entropy_mesh

settings_file.export_to_xml()

# plot setting 
plot = openmc.Plot(plot_id=1)
plot.origin = [0, 0, 0]
plot.width = [21.50, 21.50]
plot.pixels = [2000, 2000]
plot.color_by = 'material'

# Instantiate a Plots object and export to XML
plot_file = openmc.Plots([plot])
plot_file.export_to_xml()

###############################################################################
#                     Set volumes of depletable materials
###############################################################################

# Compute cell areas
area = {}
area[fuel_pellet] = np.pi * fuel_or.coefficients['r'] ** 2

# Set materials volume for depletion. Set to an area for 2D simulations
fuel_21.volume = area[fuel_pellet]
fuel_31.volume = area[fuel_pellet] * 264 
fuel_36.volume = area[fuel_pellet]
fuel_46.volume = area[fuel_pellet]
gad.volume     = area[fuel_pellet]
ifba.voume     = area[fuel_pellet]
###############################################################################
#                   Initialize and run depletion calculation
###############################################################################


###############################################################################
#                   Initialize and run depletion calculation
###############################################################################

# op = openmc.deplete.Operator(geometry, settings_file, chain_file, diff_burnable_mats=True)

# Perform simulation using the predictor algorithm
# openmc.deplete.integrator.cecm(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.predictor(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.cecm(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.epc_rk4(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.leqi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.celi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.si_celi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.si_leqi(op, time_steps, power_density=power_den)

# Get fission Q values from JSON file generated by get_fission_qvals.py
with open('/home/jiankai/depletion-comparison/data/depletion/serpent_fissq.json', 'r') as f:
    serpent_fission_q = json.load(f)

# Set up depletion operator
chain_file = '/home/jiankai/depletion-comparison/data/depletion/chain_casl_pwr.xml'
op = openmc.deplete.Operator(geometry, settings_file, chain_file, diff_burnable_mats=True, 
    fission_q=serpent_fission_q,
    fission_yield_mode="average")
    
integrator = openmc.deplete.CELIIntegrator(op, time_steps, power_density=power_den)

integrator.integrate()
