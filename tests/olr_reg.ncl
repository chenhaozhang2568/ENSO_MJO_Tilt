load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
begin
;************************************************
; open file and read in variable
;************************************************

in3  = addfile("RMM_pattern_stander.nc","r")
in4  = addfile("MJO_PC_INDEX_stander.nc","r")

pc11 = in4->PC1
pc22 = in4->PC2
lon = in3->lon
time = in4->time

in5  = addfile("/data2/home/hufeng523/mission3/data/olr.nc","r")
olr_ys = in5->olr(0:12418,:,:)
lat = in5->lat

olr_reg1 = regCoef_n(pc11,olr_ys,0,0)
olr_reg2 = regCoef_n(pc22,olr_ys,0,0)
printVarSummary(olr_reg1)
;olr_reg = olr_reg1
;olr_reg = olr_reg1 + olr_reg2
;printVarSummary(olr_reg)
;************************************************
; create BandPass Filter

olr_reg = new((/12419,121,240/), float)
do i = 0,12418
do j = 0,239
do k = 0,120
olr_reg(i,k,j) = (olr_reg1(k,j)*pc11(i)/0.236+olr_reg2(k,j)*pc22(i)/0.221)/2.
end do
end do
end do

olr_reg!0 = "time"
olr_reg!1 = "lat"
olr_reg!2 = "lon"
olr_reg&time = time
olr_reg&lat = lat
olr_reg&lon = lon
printVarSummary(olr_reg)

diro = "./"
  filo = "olr_RMM_reg2.nc"
  system("/bin/rm -f "+diro+filo)   ; remove any pre-existing file
  ncdf = addfile(diro+filo,"c")     ; open output netCDF file
                                    ; make time an UNLIMITED dimension 
  filedimdef(ncdf,"time",-1,True)   ; recommended  for most applications
                                    ; output variables directly  
  ncdf->olr_reg       = olr_reg 
end
