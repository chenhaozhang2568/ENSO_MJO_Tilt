load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"  
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"  
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl" 

begin

;***********************************************************
; Open PC components file created in 'mjo_14.ncl'
;***********************************************************
 in3  = addfile("w_reg_shiyan.nc","r")
omega2 = in3->w
lev = in3->lev
lat = in3->lat
printVarSummary(omega2)


  lon2 = in3->lon
   lon = fspan(0,360,1440)
   knots = 9
   
   omega = csa1(lon2,omega2,knots,lon)   
   omega!3 = "lon"
   omega&lon = lon
   lon@units = "degrees_east"
   omega!1 = "lev"
   omega&lev = lev
   omega!2 = "lat"
   omega&lat = lat
   printVarSummary(omega)
  
  
  ;olr_mjo    =  dim_avg_n_Wrap(omega(:,:,{-15:15},:),(/1,2/))
  ;printVarSummary(olr_mjo)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;	
     mjo =asciiread("zong_center_lon.txt",(/1281/), "integer")  
 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;	
  
  
  omega_d    =  dim_avg_n_Wrap(omega(:,{1000:600},{-15:15},:),(/1,2/))
  omega_u    =  dim_avg_n_Wrap(omega(:,{400:200},{-15:15},:),(/1,2/))
  omega_mjou    =  smth9_Wrap(omega_u, 0.50,  0.25, True) ;round(omega_u * 10^10,3)
  omega_mjod    =  smth9_Wrap(omega_d, 0.50,  0.25, True);round(omega_d * 10^10,3)
copy_VarMeta(omega_d,omega_mjod) 
 copy_VarMeta(omega_d,omega_mjou) 
  printVarSummary(omega_mjou)
   
  omegaa = dim_avg_n_Wrap(omega(:,:,{-15:15},:),2)
  
  
  y = 1281
  stren = new( (/y/), double ) 
  stren_u = new( (/y/), double )   ; 1614
  stren_d = new( (/y/), double )  
  posi_mjo = new( (/y/), integer )  
  posi_u = new( (/y/), integer )
  posi_d = new( (/y/), integer )
  guoduu = new( (/2400/), integer )
  guoduu = -1
  guodud = new( (/2400/), integer )
  guodud = -1
  fanwei_c = new( (/y/), integer )
  fanwei_mjo = new( (/y/), integer )
    speed = new( (/y/), double )
 
 omega_mjou2 = new((/1281,1680/),"double")
omega_mjou2(:,240:1679)=omega_mjou(:,:)
omega_mjou2(:,0:239)=omega_mjou(:,1200:1439)
  omega_mjod2 = new((/1281,1680/),"double")
omega_mjod2(:,240:1679)=omega_mjod(:,:)
omega_mjod2(:,0:239)=omega_mjod(:,1200:1439)
 
  do i = 0, y-1  
  guodud = 0
  guoduu = 0
  posi_mjo(i) = mjo(i) + 240
  ;speed(i) = dim_avg_n_Wrap(dmdtave(i,posi_mjo(i)+16:posi_mjo(i)+33),0) - dim_avg_n_Wrap(dmdtave(i,posi_mjo(i)-20:posi_mjo(i)-4),0)
  x = posi_mjo(i)
  
  m = 0
  do j1 = 0, x
  if (omega_mjod2(i,j1) .ge. 0) then;(omega_mjod(i,j1) .le. 0 .and. omega_mjod(i,j1) .gt. -1) then      ;omega_mjo2(i,j) .le. 0 .and. omega_mjo2(i,j) .gt. -1
  guodud(m) =  j1   ;posi_mjo(i) - j  
  m = m + 1
  end if   
  end do 
   n = 0
  do j2 = 0, x
  if (omega_mjou2(i,j2) .ge. 0);(omega_mjou(i,j2) .lt. 1 .and. omega_mjou(i,j2) .ge. 0) then      ;omega_mjo2(i,j) .le. 0 .and. omega_mjo2(i,j) .gt. -1
  guoduu(n) =  j2   ;posi_mjo(i) - j  
  n = n + 1
  end if   
  end do   
  posi_d(i) =    max(guodud)      ;posi_mjo(i) - min(guodu)       ; minind( omega_mjo(i,0:posi_mjo(i))) 
  posi_u(i) =    max(guoduu)
  fanwei_c(i) = posi_d(i)-posi_u(i)
  fanwei_mjo(i) = posi_mjo(i) - posi_d(i)
  stren(i) = dim_avg_n_Wrap(dim_avg_n_Wrap(omegaa(i,{1000:200},x-40:x+40),0),0)
  
  if (fanwei_c(i) .ge. 1.and.posi_u(i).ge.1.and.posi_d(i).ge.1) then
  stren_u(i) = dim_avg_n_Wrap(dim_avg_n_Wrap(omegaa(i,{200:500},posi_u(i)-1:posi_d(i)-1),0),0)

  stren_d(i) = dim_avg_n_Wrap(dim_avg_n_Wrap(omegaa(i,{1000:600},posi_u(i)-1:posi_d(i)-1),0),0)
  else
  stren_u(i) = 0.
  stren_d(i) = 0.
  end if   
  end do
    
      
  y2 = 1281     ; 1215
  stren_u2 = new( (/y2/), double )   ; 1614
  stren_d2 = new( (/y2/), double )  
  posi_mjo2 = new( (/y2/), double )
  posi_d2 = new( (/y2/), double )
  posi_u2 = new( (/y2/), double )
  fanwei_c2 = new( (/y2/), double )
  fanwei_mjo2 = new( (/y2/), double )
  ;speed2 = new( (/y2/), double )
  
  j = 0
  do i = 0, y-1
  if (posi_u(i) .ne. -1 .and. posi_d(i) .ne. -1) then
  stren_u2(j) = stren_u(i) 
  stren_d2(j) = stren_d(i)
  posi_mjo2(j) = posi_mjo(i)* 0.25-60.
  posi_d2(j) = posi_d(i)*0.25  
  posi_u2(j) = posi_u(i)*0.25
  fanwei_c2(j) = fanwei_c(i)*0.25
  fanwei_mjo2(j) = fanwei_mjo(i)*0.25
 ;speed2(j) = speed(i)
  j = j + 1
  end if
  end do
  
  print(j)
  ;asciiwrite("position_down_w2.txt",posi_d2)
 ; asciiwrite("position_mjo_w_1000600_025.txt",posi_mjo2(0:j-1))
   ;asciiwrite("position_u_w2.txt",posi_u2)
  
  ;asciiwrite("strength_up_025.txt",stren_u2)
  ;asciiwrite("strength_down_025.txt",stren_d2)
  
  asciiwrite("fanwei_strati_w_025.txt",fanwei_c2(0:j-1))
  asciiwrite("stren_MJO_w.txt",stren)
  ;asciiwrite("speed.txt",speed2)
  ;r = escorc(posi_mjo2,fanwei)
 ; print(r)
end  
  
  
  
  
  
  
  
  
  