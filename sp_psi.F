!***********************************************************************
!
!                         AUSWERT subroutine sp_psi
!
! Converts a psi file from an MCTDH to an exact representation in a 
! single point
! Emilio R.C. 09/2024
!***********************************************************************

      program sp_psi

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

      integer  dynbytes(4),complexsize,realsize,integersize
      integer  ilbl,chkdvr,chkgrd,chkpsi,chkdat,chkprp,i,m,f
      real*8   totmem,thrs
      logical  lerr,lascii,lrst,linwf,lwght
      common /sp_psiin/ thrs,lascii,lrst,lwght
      complex*16,allocatable :: mc(:)
      complex*8 ,allocatable :: ms(:)
      real*8    ,allocatable :: mr(:)
      integer   ,allocatable :: mi(:)
      integer   ,allocatable :: rint(:)


!-----------------------------------------------------------------------
! Initialize variables
!-----------------------------------------------------------------------
      call default
      call adefault
      call alloc_ana(3,' ')
      lascii=.false.
      lwght=.false.
      lrst=.false.
      call sp_psiinput

!-----------------------------------------------------------------------
! Files to be read
!-----------------------------------------------------------------------
      lrddvr=lpop2

!-----------------------------------------------------------------------
! Open log file channel
!-----------------------------------------------------------------------
      open (ilog,file='sp_psi.log',form='formatted')
      logisopen=.true.

!-----------------------------------------------------------------------
! set no. of comparison data sets
!-----------------------------------------------------------------------
      ncomp=2

!-----------------------------------------------------------------------
! read data from psi file or rst
!-----------------------------------------------------------------------
      chkdvr=1
      chkgrd=1
      chkpsi=1
      chkdat=1
      if (lrst) then
         irst=ipsi
         chkprp=0
         call rstinfo(linwf,lerr,chkdvr,chkgrd,chkpsi,chkprp)
         read(irst,err=998) tinit
      else
         read(ipsi) filever(ipsi)
         call rdpsiinfo(ipsi,chkdvr,chkgrd,chkpsi,chkdat)
      endif
      if (psitype .eq. 1) then
         routine='sp_psi'
         message = 'Wavefunction already in exact representation'
         call errormsg
      endif
      if (chkerr(1) .ne. 0) go to 999

!-----------------------------------------------------------------------
! read data from dvr file
!-----------------------------------------------------------------------
      if(lort) then
            open(idvr,file=dname(1:dlaenge)//'/dvr',form='unformatted',
     +           status='old',err=997)
            call dvrinfo(lerr,chkdvr)
            if (lerr) then
                    write(6,*) ' Error reading: dvr'
                    stop
            end if
            close(idvr)
      end if


!-----------------------------------------------------------------------
! save data as set 1
!-----------------------------------------------------------------------
      call savecsys(1)
      call savecpsi(1)
      call savecdat(1)

!-----------------------------------------------------------------------
! set up and save exact psi information
!-----------------------------------------------------------------------
      psitype=1
      lpsiopt(1)=.false.
      lpsiopt(2)=.false.
      lpsiopt(3)=.false.
      lpsiopt(4)=.false.
      griddim=nstate
      do m=1,nmode
         griddim=griddim*subdim(m)
      enddo

      ! Electronic basis: for exact, we always use multi-set (see
      ! EINSBAS). If the incoming WF is single-set, fix it.
      if (feb.gt.0 .and. .not.lmult) then
         lmult = .true.
         leb = .false.
         nstate = gdim(feb)
         meb = dofspf(feb)
         if (nspfdof(meb).eq.1) then
            nmode = nmode-1
         else
            nspfdof(meb) = nspfdof(meb)-1
         endif
      endif
      call einexact

      call griddat
      call psidat

! complete spfdof information with koeff and eb dofs
      fkoe=ndof+1
      mkoe=nmode+1
      if(feb.gt.0) then
         meb=nmode+2 ! cf. HEINGABE
      endif
      dofspf(fkoe)=mkoe
      spfdof(1,mkoe)=fkoe
      if(feb.gt.0) then
         dofspf(feb)=meb
         spfdof(1,meb)=feb
         nspfdof(meb)=1
      endif

      call savecsys(2)
      call savecpsi(2)
      call savecdat(2)

!-----------------------------------------------------------------------
! write exact psi information to new file
!-----------------------------------------------------------------------
      if(.not.lascii) then
         write(ipsi1) progver
         call wrpsiinfo(ipsi1)
      endif

!-----------------------------------------------------------------------
! assign pointers and sizes for required arrays
!-----------------------------------------------------------------------
      lpsi=.true.
      lpsigrd=.true.
      ltrafo= lwght
      lfft=.false.
!     is any of the degrees of freedom of fft or expDVR type?
      do f=1,ndof
         if(basis(f).eq.4 .or. basis(f).eq.5) lfft=.true.
      end do

      call zeigausw

!-----------------------------------------------------------------------
! assign memory arrays
!-----------------------------------------------------------------------
      call memausw
      mrweight=mrdim
      if(lwght) then
         mrdim=mrweight+ortdim
      else
         mrdim=mrweight+1
      endif

!-----------------------------------------------------------------------
! assign work arrays
!-----------------------------------------------------------------------
      mrworkr = mrdim
      mrdim   = mrworkr+workrdim

      miworki = midim
      midim   = miworki+workidim

      msworks = msdim
      msdim   = msworks+worksdim

!-----------------------------------------------------------------------
! call conversion routine
!-----------------------------------------------------------------------
      dynbytes(1) = complexsize()*(mcdim/8+1)
      dynbytes(2) = realsize()   *(mrdim/8+1)
      dynbytes(3) = integersize()*(midim/8+1)
      dynbytes(4) = (complexsize()/2)*(msdim/8+1)
      ilbl=0
      totmem=0.d0
      do i=1,4
         totmem=totmem+dynbytes(i)
         if( dynbytes(i).lt.0 ) then
            ilbl=1
            totmem=totmem+4.294967296d+9
         endif
      enddo
      totmem=totmem*8.d0

      call wrversion(ilog)
      write(ilog,'(a)') '  *** Memory allocated dynamically ***'
      write(ilog,'(4(a,i12,/),a,f16.0,a,f11.2,a)')
     + ' complex*16 :', mcdim,
     + ' real*8     :', mrdim,
     + ' integer    :', midim,
     + ' complex*8  :', msdim,
     + ' Total   :',totmem,' Bytes ,',totmem/1048576.d0,' MB'
      write(ilog,'(80a1,/)') ('-',i=1,80)
      call myflush(ilog)

      if(ilbl.gt.0) then
         routine='sp_psi'
         write(message,'(3a,f9.1,a)')
     +     'Integer overflow in memory assignment. ',
     +     'You are asking for too much memory.    ',
     +     'Total Memory asked for =',totmem/1048576.d0,' MB'
         call errormsg
      endif

      write(ilog,'(a,/25i4)') 'Order of DOFs in exact mode:',
     +                         (cspfdof(f,1,2),f=1,cnspfdof(1,2))
      write(ilog,*)

      allocate(mc(mcdim),ms(msdim),mr(mrdim),mi(midim))
      

c     Read indexed from the CLI

      write(6,*) "Enter the index/es for the ", nmode, "modes"
      allocate(rint(nmode))
      read(5,*) (rint(i), i=1,nmode)

      
      do i=1, nmode
         write(6,*) "Index for mode", i, ":", rint(i)
      enddo

!-----------------------------------------------------------------------
! recall psi data for length of arrays
!-----------------------------------------------------------------------
      call getcpsi(1)

      call subsp_psi(mc(mcpsi),ms(mspsi),rint,
     +     mi(mijindx),cdgldim(2),mr(mrort),mr(mrweight),
     +     mr(mrtrafo),mc(mcexphin),mc(mcrueck),mi(mifftfak))

      stop

  997 write(6,'(a,/a)') 'Cannot open dvr-file:',
     +                   dname(1:dlaenge)//'/dvr'
      stop 1
  998 continue
      write(6,'(a)') 'Error reading time from restart-file.'
      stop 1
  999 continue
      write(6,'(a)') 'Error reading psi-file.'
      stop 1
      end

!#######################################################################
!***********************************************************************
!     calculates the exact wavefunction for a single point over time
!***********************************************************************

      subroutine subsp_psi (psi,spsi,idxs,jindx,dgldim1,
     +                      ort,weight,trafo,exphin,rueck,fftfak)

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

      integer    dgldim1,i,jindx(adim),d,iz,ndof1
      integer    fftfak(fftdim),idum(1),chkdvr,m,n,f,nmode1
      integer    igpoint(maxdim),findx(maxgdim)
      integer    idxs(nmode), state
      real*8     rand_val
      real*8     time,dt,trafo(dvrdim),rdum(1),ort(ortdim)
      real*8     previous,now,spent,fcputime,fsystime,thrs
      real*8     weight(ortdim)
      complex*8  spsi(spsidim)
      complex*16 psi(dgldim1)
      complex*16 rueck(fftdim),exphin(expdim),zdum(1)
      complex*16 psip(nstate) ! single point
      character  actual*80
      external   fcputime,fsystime
      data       previous /0.0d0/
      save       previous
      logical    lascii,lrst,lwght
      common /sp_psiin/ thrs,lascii,lrst,lwght


!-----------------------------------------------------------------------
! read DVR data
!-----------------------------------------------------------------------
      if(lort) then
         chkdvr=1                ! dvrinfo read and stored in system data
         dvrdata(1) = lort       ! ort
         dvrdata(2) = ltrafo     ! trafo
         dvrdata(11)= lfft       ! exphin
         dvrdata(6) = lfft       ! rueck
         dvrdata(7) = lfft       ! fftfak
         call rddvr(ort,trafo,rdum,rdum,zdum,rueck,fftfak,exphin,
     +              zdum,idum,idum,rdum,chkdvr)
      endif

      call getcsys(1)
      call getcpsi(1)

!-----------------------------------------------------------------------
! Determine DVR weights, if required. Note sqrt of weights is returned
!-----------------------------------------------------------------------
      if(lwght) call getweight(trafo,weight,ort,.false.)

!-----------------------------------------------------------------------
! calculate time step for data
!-----------------------------------------------------------------------
      iz = 0
      dt   = out2/fs*step
      time = tinit/fs
! --- skip the first nskip WFs
      do i=1,nskip
         call rdpsi(ipsi,psi,spsi,jindx)
         if (lend) goto 77
         time=time + out2/fs
      enddo

! --- Loop over wave functions
   10 continue
!-------------------CALL PSI_P------------

         call psi_sp(psip, spsi, idxs, lrst, ipsi)
      if (lend) goto 77

! determine actual time and CPU-time spent since last call
      call fzeit(actual)
      i = index(actual,':')
      actual=actual(i-9:i+6)
      now=fcputime(0)+fsystime(0)
      spent=now-previous
      previous=now

      write(ilog,'(a,f8.3,a,f15.1,f15.2,4x,a16)')
     +     'Writing WF : ',time,' fs, ',now,spent,actual
      call myflush(ilog)
      call getcsys(2)
      call getcpsi(2)
! --- Handle multi-set electronic states
      nmode1=nmode
      if(nstate.gt.1 .and. lort) nmode1=nmode+1
      
! "coefficients"
      if(lascii) then
         write(ipsi1,'(1x/,a,f9.4,/)') '# Time =', time
         i=0
         do m=1,nmode1
            do n=1,nspfdof(m)
               f=spfdof(n,m)
               i=i+1
               findx(i) = f
            enddo
         enddo
         ndof1=i  ! this is necessary to remove dummy DOFs    

         write(ipsi1,'(a,8x,a,3x,a,4x,a,12x,55(a9,5x))')
     +     '#    Re(WF)','Im(WF)',' abs(WF)',' abs2(WF)',
     +     (modelabel(findx(i)),i=1,ndof1)
         

         call initvxi(igpoint,maxdim) ! Set igpoint(i)=1
         igpoint(1)=0
               
! ---    determine set of grid-points
         i=0
         do m=1,nmode1
         do n=1,nspfdof(m)
               f=spfdof(n,m)
               i=i+1
               if (igpoint(i).lt.gdim(f)) then
                  igpoint(i)=igpoint(i)+1
                  goto 15
               else
                  igpoint(i)=1
               endif
            enddo     ! n=1,nspfdof(m)
         enddo        ! m=1,nmode1
  15     continue
! ---    remove weights from WF if required
         if(lwght) then   ! if(findx(i).ne.feb) then
            do i=1,ndof1
               if(findx(i).ne.feb) then
                  psip(2) = psip(2)/
     +            weight(zort(findx(i))+idxs(i)-1)
               endif
            enddo
         endif
! ---    write WF and grid points
                  ! write(6,*) "Here"
         do state=1,nstate
            write(ipsi1,996) psip(state),cdabs(psip(state)), 
     +           cdabs(psip(state))**2,
     +           (ort(zort(findx(i))+idxs(i)-1),i=1,ndof1)
         enddo
      else    ! not ascii
            do state=1,nstate
               write(ipsi1) psip(state),cdabs(psip(state)), 
     +              cdabs(psip(state))**2,
     +             (ort(zort(findx(i))+idxs(i)-1),i=1,ndof1)
            enddo
         endif
         

      time=time + dt
      iz=iz+1
      if(lrst) goto 77
      if(iz.ge.ncut) goto 77

      call getcsys(1)
      call getcpsi(1)

!-----------------------------------------------------------------------
! if more data, go to beginning of loop
!-----------------------------------------------------------------------
      do i=2,step
         call rdpsi(ipsi,psi,spsi,jindx)
         if (lend) goto 77
      enddo

      goto 10

 77   continue
      close(ipsi)
      close(ipsi1)

      return
 991  format(6f13.8)
 992  format(3(2f13.8,2x))
 995  format(f17.12,1x,33f14.5)
 996  format(2f14.8,2x,33f14.5)
      end

!-----------------------------------------------------------------------
!***********************************************************************
!
! sp_psiINPUT
!
! get input parameters and file names for calculation
!
!-----------------------------------------------------------------------
      subroutine sp_psiinput


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

      integer        iarg, myiargc, ierr,i
      real*8         thrs
      character*80   buf
      character*(c5) filename
      character*12   oformat
      logical        lfileout,lascii,lrst,lwght
      common /sp_psiin/ thrs,lascii,lrst,lwght
!-----------------------------------------------------------------------

      nskip=0
      ncut = 999999
      thrs=-1.d0
      filein  = './'
      fileout  = 'sp_psi.ex'
      lfileout=.false.
      iarg = myiargc()
      if (iarg .lt. 1) goto 200

      i = 0
 100  i=i+1
      call mygetarg(i,buf)
      if( buf(1:1) .eq. '-' ) then
         if( buf(1:3) .eq. '-f ' .or.  buf(1:3) .eq. '-i ') then
            i = i+1
            call mygetarg(i,buf)
            filein = buf
         else if( buf(1:3) .eq. '-o ' ) then
            i = i+1
            call mygetarg(i,buf)
            fileout = buf
            lfileout=.true.
         else if( buf(1:5) .eq. '-stp ' ) then
            i = i+1
            call mygetarg(i,buf)
            read(buf,*,iostat=ierr) step
            if (ierr .ne. 0  .or.  step .le. 0) then
               write(6,'(2a/,a)')
     +           ' The  argument following the option -n',
     +           ' must be a positive integer. Not : ', buf
               stop
            end if
         else if( buf(1:6) .eq. '-thrs ' ) then
            i = i+1
            call mygetarg(i,buf)
            read(buf,*,iostat=ierr) thrs
            if (ierr .ne. 0) then
               write(6,'(2a/,a)')
     +           ' The  argument following the option -thrs',
     +           ' must be a real number. Not : ', buf
               stop
            end if
         else if( buf(1:3) .eq. '-n ') then
            i = i+1
            call mygetarg(i,buf)
            read(buf,*,iostat=ierr) ncut
            if (ierr .eq. 1) then
               write(6,'(/a)') 'Cannot read argument of -n'
               stop
            endif
         else if( buf(1:5) .eq. '-skip ') then
            i = i+1
            call mygetarg(i,buf)
            read(buf,*,iostat=ierr) nskip
            if (nskip .lt. 0) then
               write(6,'(/a)')
     +         'Argument of -skip must be positive integer'
               stop
            endif
            if (ierr .eq. 1) then
               write(6,'(/a)') 'Cannot read argument of -skip'
               stop
            endif
         else if( buf(1:3) .eq. '-w ' ) then
            lovwr=.true.
         else if( buf(1:4) .eq. '-W  ' ) then
            lwght=.true.
         else if( buf(1:6) .eq. '-ort  ' ) then
            lort=.true.
            lascii=.true.
         else if( buf(1:6) .eq. '-ascii' ) then
            lascii=.true.
         else if( buf(1:6) .eq. '-rst  ' ) then
            lrst=.true.
         else if( buf(1:3) .eq. '-h ' .or. buf(1:3) .eq. '-? ' ) then
            write(6,'(78a1)')  ('-',i=1,78)
            write(6,'(1x,a,7(/10x,a))')
     +   'Purpose: Evaluates the wavefunction for a given set of ',
     +    'grid indices to numerically exact form. ',
     +   'To inspect the WF, the output may also be in ASCII format',
     +   'with or without printing coordinates (option -ort).',
     +   'Default is showing the "naked" DVR populations, but',
     *   'DVR-weights can be removed from the WF by setting the ',
     +   'option -W (requires that -ort is also set). NB: -W does',
     +   'the opposite of -nw used in other analysis routines.'

            write(6,'(/1x,2a,/19x,a)') 'Usage  : sp_psi86 ',
     +      '[-f -i -o -stp -n -skip -thrs -ort -ascii -W',
     +      ' -rst -w -dp -ver -h -?]'
            write(6,'(/,a)') 'Options : '
            write(6,'(2a,/,9x,a)')
     +               ' -f FILE : The wavefunction is read '
     +              ,'from file FILE rather than from ./psi'
     +      ,'  The string FILE may be a relative or a full path-name.'
            write(6,'(a)') ' -i FILE : Same as -f FILE.'
            write(6,'(2a,/,11x,a)')
     +               ' -o FILE : The output is written to file ',
     +               'FILE rather than to ./psi.ex',
     +       'The string FILE may be a relative or a full path-name.'
            write(6,'(2a)')
     +      ' -rst    : A restart file rather than a psi file is read',
     +                  ' as input.'
            write(6,'(2a)') ' -stp step: Compute the exact WF',
     +                      ' only every step-th output.'
            write(6,'(2a)')
     +               ' -n num  : only the first num WFs are read '
     +              ,' from psi-file.'
            write(6,'(a)')
     +               ' -skip m : the first m WFs are skipped.'
            write(6,'(2a,/11x,a)')
     +      ' -ort    : The WF and their coordinates are written to',
     +                 ' output file.',
     +                 'Note that -ort sets -ascii.'
            write(6,'(a,/11x,a,/11x,a)')
     +      ' -W      : The DVR-weights are removed from the WF.',
     +                'Useful for plotting the WF.',
     +                '-W requires that -ort is set, otherwise ignored.'
            write(6,'(a,/11x,a)')
     +      ' -thrs thrs : The WF and their coordinates are written',
     +              'only if(abs(WF).GT.thrs). Works only with -ort.'
            write(6,'(2a,/11x,a,/11x,a)')
     +      ' -ascii  : The output file is in ascii rather than binary',
     +                 ' format.',
     +                 'In this case the (dummy) A-vector and psiinfo',
     +           'are not written, i.e. the file contains the WF only.'
            write(6,'(2a)')
     +               ' -w      : An existing output file is ',
     +               'overwritten.'
            write(6,'(a)')
     +               ' -ver    : Version information about the program.'
            write(6,'(a,/a)')
     +               ' -h      : Print this help text.',
     +               ' -?      : Print this help text.'
            write(6,'(78a1)')  ('-',i=1,78)
            stop
         else if( buf(1:4) .eq. '-ver' ) then
            call wrversion(6)
            stop
         else
            write(6,'(2a)') ' unknown option : ', buf
            stop
         end if
         if (iarg .le. i) goto 200
         goto 100
      else
         write(6,'(2a)') ' unknown argument : ', buf
         stop
      end if

 200  continue
      if(lwght.and. .not.lort) write(6,'(a)')
     + 'WARNING: Option -W is ignored because -ort is not set!'
      lwght=lwght.and.lort
      laein=index(filein,' ')-1
      if(laein.gt.6) then
         if (filein(laein-6:laein) .eq. 'restart' ) lrst = .true.
         if (filein(laein-2:laein) .eq. 'psi' )     lrst = .false.
      endif
      if (filein(laein:laein) .eq. '/') then
         if(lrst) then
            irst = ipsi
            filein = filein(1:laein)//'restart'
            laein  = laein + 7
         else
            filein = filein(1:laein)//'psi'
            laein  = laein + 3
         endif
      end if
      call abspath(filein,laein)

      laeout=index(fileout,'  ')-1
      call abspath(fileout,laeout)

!-----------------------------------------------------------------------
!  Opening the files.
!-----------------------------------------------------------------------
      write(6,'(80a)') ('*',i=1,80)
      filename=filein(1:laein)
      open(ipsi,file=filename,form='unformatted',status='old',
     +     err=1000)
      write(6,'(2a)') ' The inputfile is  : ',filein(1:laein)

      oformat='unformatted'
      if(lascii) oformat='formatted'

      filename=fileout(1:laeout)
      if (lovwr) then
         open (ipsi1,file=filename,form=oformat,status='unknown',
     +           err=1000)
      else
         open (ipsi1,file=filename,form=oformat,status='new',
     +           err=1010)
      endif
      write(6,'(2a)') ' The outputfile is : ',fileout(1:laeout)
      write(6,'(2a)') ' The log file is   :  ./sp_psi.log'


      return

 1000 write(6,*)' Error opening file : '
      write(6,*) filename
      stop
 1010 write(6,*)' File already exists: '
      write(6,*) filename
      stop

      end