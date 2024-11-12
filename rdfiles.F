C***********************************************************************
C
C  ANALYSE programs module  RDFILES
C
C  This file contains routines to read the files output by the
C  MCTDH program
C
C rdpsi:  read the wavefunction from the psi file
C rdpsigrid: read the wavefunction from the psi file and transform it
C            to the full grid (primitive basis) representation.
C rdgridpop: read the gridpop file
C
C
C***********************************************************************

C***********************************************************************
C
C                RDPSI
C
C Called by the various ANALYSE programs in order to read the
C psi file data which is formatted differently depending
C on selected options.
C
C***********************************************************************

      subroutine rdpsi(unit,psi,spsi,jindx)

      implicit none

#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"

      integer    unit,dgl,jindx(*)
!      integer    actblock(maxsta),dgldim1,zeig,s
      complex*8  spsi(dgldim)
      complex*16 psi(dgldim)
      logical    lpsisp,lpsicm,lerr,lselect
      integer    m,s
      lend = .false.
      lpsisp=lpsiopt(1)
      lpsicm=lpsiopt(2)
      lselect=lpsiopt(4)
      lerr = .true.

C-----------------------------------------------------------------------
C read indices for compact output wavefunction
C-----------------------------------------------------------------------
      if (lpsicm) then
         routine='rdpsi'
         message='psi in compact form is not supported'
         call errormsg
!         dgldim1=dgldim
!         zeig=0
!         do s=1,nstate
!            read (unit,end=10) actblock(s)
!            read (unit,end=10) (jindx(zeig+dgl),dgl=1,actblock(s))
!            zeig=zeig+actblock(s)
!            dgldim1=dgldim1 - (block(s)-actblock(s))
!         enddo
      else if (lselect) then
         routine='rdpsi'
         message='Selected CI is not supported'
         call errormsg
!         do s=1,nstate
!            read (unit,end=10) (jindx(zpsi(s)+dgl),dgl=1,block(s))
!         enddo
      endif

C-----------------------------------------------------------------------
C read wavefunction
C-----------------------------------------------------------------------
CDWF  If dynamical WF: read extra info about the wavefunction
      if (psitype.eq.10) then
         read (unit,end=10,err=30) dgldim,
     +   ((dim(m,s),m=1,nmode),s=1,nstate)
         call psidat
      endif
      if (lpsisp) then
         read (unit,end=10,err=30) (spsi(dgl),dgl=1,dgldim)
         do dgl=1,dgldim
            psi(dgl)=spsi(dgl)
         enddo

!      else if (lpsicm) then
!         read (unit,end=10,err=30) (spsi(dgl),dgl=1,dgldim1)
!
!         zeig=1
!         do s=1,nstate
!            call cmavec(psi(zpsi(s)),spsi(zeig),jindx(zeig),
!     +           block(s),actblock(s))
!            zeig=zeig+actblock(s)
!         enddo
!
!         do dgl=adim+1,dgldim
!            psi(dgl)=spsi(zeig)
!            zeig=zeig+1
!         enddo

      else
         read (unit,end=10,err=30) (psi(dgl),dgl=1,dgldim)

      endif
      goto 20

 10   lend=.true.
 20   continue

      return

 30   write(6,*) 'ERROR in rdpsi (source/analyse/rdfiles.F).'
      write(6,*) 'Error occurred while reading channel ',unit
      write(6,*) 'psi : unit=8;  psi1 : unit=48'
      write(6,*) 'dgl =', dgl,',  dgldim =', dgldim
      write(6,*) 'lpsicm, lpsisp :',lpsicm, lpsisp
      stop 1

      end

C#######################################################################
C
C CMAVEC
C
C copies the truncated A-vector, stored in compact storage mode,
C to the psi vector, filling in blank spaces with 0
C
C#######################################################################

      subroutine cmavec(psi,spsi,jindx,block,actblock)

      implicit none

      integer block,actblock,jindx(actblock),
     +        b1,b2
      complex*8  spsi(actblock)
      complex*16 psi(block)

      if (actblock .eq. 0) then
         call zerovxz(psi,block)
         return
      endif

      b1=1
      do b2=1,block
         if (b2 .eq. jindx(b1)) then
            psi(b2)=spsi(b1)
            if (b1 .lt. actblock) b1=b1+1
         else
            psi(b2)=0.0d0
         endif
      enddo

      return
      end

C***********************************************************************
C
C                RDPSIGRID
C
C Called by the various ANALYSE programs in order to read the
C psi file data, and transform the wavefunction to the full grid
C (primitive basis) representation
C
C***********************************************************************

      subroutine rdpsigrid(unit,psi,spsi,jindx,workc,lrst)

      implicit none

#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"

      integer    unit,zeig1,zeig2,zgrd,m,s,jindx(adim)
      integer    swapzeig,vdim1,i
      complex*8  spsi(dgldim)
      complex*16 psi(griddim),workc(dgldim+2*(griddim/nstate))
      logical    lrst

      ! Check if the workspace dimension is large enough
      if(dgldim+2*(griddim/nstate).gt.workcdim) then
         routine='rdpsigrid'
         message='workcdim is too small'
         write(6,*) 'workcdim =                 ', workcdim
         write(6,*) 'dgldim+2*(griddim/nstate) =',
     +               dgldim+2*(griddim/nstate)
         call errormsg
      endif
      ! zeig1 points after the wf data
      zeig1=dgldim+1
      ! zeig2 points after the zeig1 by an offset of griddim/nstate
      zeig2=zeig1+griddim/nstate


! Read mctdh wavefunction into array spsi (single precision psi)
      if(lrst) then
         read(unit,err=999) (workc(i),i=1,dgldim)
      else
         call rdpsi(unit,workc,spsi,jindx)
      endif
      zgrd=1
      do s=1,nstate
! For each state s, the A-vector coeff are copied to the workc
         call cpvxz(workc(zpsi(s)),workc(zeig2),block(s))
         vdim1=1
         do m=1,nmode
! For each mode m in state s, a tensor mutiplication is done
! and the A-vector coeff and SPFs are combined to obtain the wf
! in the grid representation
           ! Swap pointers for buffers
            swapzeig=zeig1
            zeig1=zeig2
            zeig2=swapzeig

            ! Use mtxxzz function for matrix-tensor multiplication
            call mtxxzz (workc(zetf(m,s)),workc(zeig1),
     +           workc(zeig2),vdim1,dim(m,s),ndim(m,s),subdim(m))
            ! Update dimension multiplier by grid points in m
            vdim1=vdim1*subdim(m)
            ! do i=1,subdim(m)+1
            !    write(6,*) "grid:", i, "psi:", workc(i)
            ! enddo
         enddo
! Copy result to the appropriate grid position in psi
         call cpvxz(workc(zeig2),psi(zgrd),griddim/nstate)
! zgrid now points at the starting index for the next state
         zgrd=zgrd+griddim/nstate
      enddo

      return

 999  write(6,*) ' ERROR reading restart file (-rst)'
      stop

      end

!***********************************************************************
!
!                PSI_SP
!
! Called by the various ANALYSE programs to read the
! psi file data and calculate the wavefunction at a single point of
! the grid (primitive basis) for several electronic states.
! 
! EMILIO R.C. 09/2024
!***********************************************************************
      subroutine psi_sp(psi_value, psi, grid_indices, lrst, unit)

      implicit none

   ! Include the global variables
#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"

   ! Declare parameters and variables
      integer unit
      logical lrst
      integer grid_indices(nmode)
      complex*8  spsi(dgldim)
      complex*16 psi_value(nstate)
      complex*16 psi(dgldim)
      complex*16 workc(dgldim+2*(griddim/nstate))

   ! Local variables
      integer m, s, i, idx, config_index
      integer jindx(adim), spf_start, a_start
      integer spf_indices(nmode)
      complex*16 spf_values(nmode, maxspf)
      complex*16 a_coeff, product
      integer total_configs
      integer dima(nmode)
      integer indices(nmode)
      complex*16 psi_work(dgldim)

   ! Initialize psi_value to zero
      do s = 1, nstate
         psi_value(s) = (0.0d0, 0.0d0)
      end do

   ! Read the wavefunction into psi
      if(lrst) then
         read(unit,err=999) (workc(i),i=1,dgldim)
      else
         call rdpsi(unit,workc,spsi,jindx)
      endif

   ! Loop over states
      do s = 1, nstate
   ! Extract SPF values at the specified grid indices for state 's'
         do m = 1, nmode
         spf_start = zetf(m, s)
   ! Ensure grid index is within valid range
         if (grid_indices(m) < 1 .or. grid_indices(m) > subdim(m)) then
            write(6,*) "Error: grid_indices(", m, ") out of bounds."
            stop
         endif
         do i = 1, dim(m, s)
   ! Calculate the index for the i-th SPF at the specific grid point
            idx = spf_start + (i-1)*subdim(m) + grid_indices(m) - 1
            spf_values(m, i) = workc(idx)
         end do
         end do

   ! Compute the wf value for state s
         psi_value(s) = (0.0d0, 0.0d0)
   ! Set number of A-vec coeff for state s
         total_configs = block(s)

   ! Loop over A-vector
         a_start = zpsi(s)
         do config_index = 1, total_configs
         idx = a_start + config_index - 1
         a_coeff = workc(idx)

   ! Decode the configuration indices
         call deco_conf(config_index, dim(1:nmode, s), indices)

   ! Compute the product of SPF values for this configuration
         product = (1.0d0, 0.0d0)
         do m = 1, nmode
            i = indices(m)
            product = product * spf_values(m, i)
         end do

   ! Accumulate psi_value(s)
         psi_value(s) = psi_value(s) + a_coeff * product
         end do
      end do

   ! Output the value of psi at the grid point
      do s = 1, nstate
         write(6, *) 'The value of psi for state ', s
         write(6, *)  dble(psi_value(s)), dimag(psi_value(s))
      end do

      return

999   write(6, *) 'ERROR reading wavefunction'
      stop

      end

!***********************************************************************
!                    decode_configuration
! Purpose: 
!   This subroutine converts a 1D linear index (config_index) into 
!   a set of multidimensional indices (indices) corresponding to 
!   each mode in an MCTDH wavefunction.
!
! Inputs:
!   config_index - The 1D index representing a configuration.
!   dima         - An array containing the number of SPFs for each mode.
!
! Output:
!   indices      - The decoded indices corresponding to each mode, 
!                  based on the configuration index and mode dimensions.
!
! Description:
!   The subroutine works by dividing the linear config_index by the 
!   product of the mode dimensions of the higher modes and computes the 
!   remainder, which is used to calculate the indices for the lower modes.
!
! EMILIO R.C. 09/2024
!***********************************************************************

      subroutine deco_conf(config_index, dima, indices)

      implicit none
#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"

      integer config_index
      integer dima(nmode)
      integer indices(nmode)

      integer m, remainder, divisor, i

! Initialize the remainder as config_index - 1 
! because Fortran indices start at 1.
      remainder = config_index - 1

! Loop over modes, from the last mode to the first (reverse order)
      do m = nmode, 1, -1
         divisor = 1
! Compute the product of dims(m+1), dims(m+2), ..., dims(nmode)
         if (m .lt. nmode) then
            do i = m + 1, nmode
               divisor = divisor * dima(i)
            end do
         end if
! Calculate the index for mode m by dividing remainder by divisor
         indices(m) = remainder / divisor + 1
! Update remainder for the next mode by subtracting the part already decoded
         remainder = remainder - (indices(m) - 1) * divisor
      end do

      return
      end



C***********************************************************************
C
C                RDGRIDPOP
C
C Called by the various ANALYSE programs in order to read the
C gridpop file data
C
C***********************************************************************


      subroutine rdgridpop(unit,fgpop,gpop1,gpop2)

      implicit none

#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"

      integer unit,g,s,m,n,f,ggdim
      real*4 fgpop(2*maxgdim)
      real*8 gpop1(ortdim,nstate),gpop2(ortdim,nstate)

      lend=.false.

      if (lgpel .or. nstate .eq. 1) then
         do s=1,nstate
            do m=1,nmode
               do n=1,nspfdof(m)
                  f=spfdof(n,m)
                  ggdim = gdim(f)
                  if(basis(f).eq.12) ggdim = ipbaspar(3,f)
                  read(unit,err=900,end=999)
     +                 (fgpop(g),g=1,gdim(f))
     +               ,(fgpop(g),g=gdim(f)+1,gdim(f)+ggdim)
                  do g=1,gdim(f)
                     gpop1(zort(f)-1+g,s)=fgpop(g)
                  enddo
                  do g=1,ggdim
                     gpop2(zort(f)-1+g,s)=fgpop(gdim(f)+g)
                  enddo
               enddo
            enddo
         enddo
      else
         do m=1,nmode
            do n=1,nspfdof(m)
               f=spfdof(n,m)
               ggdim = gdim(f)
               if(basis(f).eq.12) ggdim = ipbaspar(3,f)
               read(unit,err=900,end=999)(fgpop(g),g=1,gdim(f)+ggdim)
               do g=1,gdim(f)
                  gpop1(zort(f)-1+g,1)=fgpop(g)
               enddo
               do g=1,ggdim
                  gpop2(zort(f)-1+g,1)=fgpop(gdim(f)+g)
               enddo
            enddo
         enddo
      endif

      go to 20
  999 lend=.true.
   20 continue

      return
  900 routine='Rdgridpop'
      write(message,'(a,i5)' )
     +   'Error reading gridpop file on channel : ',unit
      call errormsg
      end

C***********************************************************************
C
C                RDCHK
C
C Called by the various ANALYSE programs in order to read the
C check file data
C
C the array lrdchk(3) enables storage of data in various arrays
C  (see aglobal.inc). See analyse/statepop.F for example.
C
C if lrdchk(n)=.true., following is read and stored
C lrdchk(1) state populations -> spop, energies -> euncorr, etot
C lrdchk(2) natural orbital populations -> dicht3
C lrdchk(3) properties -> prop
C
C Arrays specified by lrdchk must be correctly assigned in calling
C routine.
C
C***********************************************************************

      subroutine rdchk(unit,ecorr,etot,spop,dicht3,prop)

      implicit none

#include "parameter.inc"
#include "global.inc"
#include "aglobal.inc"
#include "griddat.inc"
#include "psidef.inc"
#include "daten.inc"
#include "compdat.inc"
#include "compdat1.inc"
#include "channels.inc"

      integer unit,i,s,m,n,f
      real*8 spop(nstate),ecorr(nstate),etot(nstate),
     +       dicht3(d3matdim),prop(maxdim,maxsta,maxprop),
     +       q,dq,expn,dexpn

C-----------------------------------------------------------------------
C check maxprop large enough to read properties written in check file
C-----------------------------------------------------------------------
      if (maxprop .lt. 4) then
         routine='Rdchk'
         message='Increase MAXPROP. Minimum : 4'
         call errormsg
      endif

C-----------------------------------------------------------------------
C read state populations
C-----------------------------------------------------------------------
      if (lrdchk(1)) then
         read(ichk,err=900,end=999)
     +       (spop(s),ecorr(s),etot(s),s=1,nstate)
      else
         read(ichk,err=900,end=999)
      endif

C-----------------------------------------------------------------------
C read natural orbital populations
C-----------------------------------------------------------------------
      if (lrdchk(2)) then
         do s=1,nstate
            do m=1,nmode
                read(ichk,err=900,end=999)
     +             (dicht3(d3mat(m,s)+i),i=0,dim(m,s)-1)
            enddo
         enddo
      else
         do s=1,nstate
            do m=1,nmode
               read(ichk,err=900,end=999)
            enddo
         enddo
      endif

C-----------------------------------------------------------------------
C read other properties
C-----------------------------------------------------------------------
c -----  read <q>,<dq>,<n> and <dn>
      if (filever(ichk).ge.8.2002d0) then
         do s=1,nstate
            do m=1,nmode
               do n=1,nspfdof(m)
                  f=spfdof(n,m)
                  if (ldvr(f) .or. basis(f) .eq. 4 .or.
     +                 basis(f) .eq. 6) then
                     read(ichk,end=999,err=900) q,dq,expn,dexpn
                  else
                     q=0.0d0
                     dq=0.0d0
                     expn=0.0d0
                     dexpn=0.0d0
                  endif
                  if (lrdchk(3)) then
                     prop(f,s,1)=q
                     prop(f,s,2)=dq
                     prop(f,s,3)=expn
                     prop(f,s,4)=dexpn
                  endif
               enddo
            enddo
         enddo
      endif

      go to 20
  999 lend=.true.
   20 continue

      return
  900 routine='Rdchck'
      write(message,'(a,i5)' )
     +   'Error reading check file on channel : ',unit
      call errormsg
      end

