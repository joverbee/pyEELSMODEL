from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.components.MScatter.mscatter import Mscatter
import numpy as np


class MscatterFFT(Mscatter):
    """
    Mutiple scattering using FFT (e.g. to concolve model with LL spectrum)
    """

    def __init__(self, specshape, llspectrum, use_padding=True):
        super().__init__(specshape, llspectrum)
        self._setname("Multiple scattering (FFT)")
        self.setdescription("Convolution of the HL spectrum with LL using fast fourier transform convolution.\nThis simulates the effect of multiple scattering\nwhich is an important effect in experimental spectra ")
        self.padding = llspectrum.size
        self.use_padding = use_padding

        #this component has no parameters, we just need to remember the low loss 
        #spectrum
        #but how do we then load that spectrum if we reload the model?
        #each component should now how to save itself so that it can reload itself
        #for most of components this can be the generic save and load defined in component
        #and some components can override it



    def calculate(self):
        if self.use_padding:
            self.calculate_padding()
        else:
            self.calculate_raw()


    def calculate_raw(self):
        fmodel=np.fft.rfft(self.data) #real fourier transform the model
        self.llspectrum.normalise()
        fll=np.fft.rfft(self.llspectrum.data) #real fourier transform the ll spectrum
        #shift zl peak position to 0!
        zlindex=self.llspectrum.getmaxindex() #need to compensate for zl peak not being at pix 0
        self.data=np.roll(np.fft.irfft(fmodel*fll),-zlindex)

    def calculate_padding(self):
        """
        Function which adds the zero padding to remove the intensity of the end of the model to come into
        the beginning of the model

        :return:
        """

        fmodel=np.fft.fft(np.pad(self.data, pad_width=(self.padding, self.padding)))#real fourier transform the model
        self.llspectrum.normalise()
        llpad = np.pad(self.llspectrum.data, pad_width=(self.padding, self.padding))
        fll=np.fft.fft(llpad) #real fourier transform the ll spectrum

        # conv = np.real(np.fft.ifft(fmodel*fll)[self.padding:-self.padding])
        conv = np.real(np.fft.ifft(fmodel*fll))

        #shift zl peak position to 0!
        zlindex=np.argmax(llpad) #need to compensate for zl peak not being at pix 0
        self.data=np.roll(conv,-zlindex)[self.padding:-self.padding]



 
'''
  //make sure the spectrum starts with y=0 and ends with y=0
  //use linear interpolation between both end
  //this is required when CCD problems occured
  //normally this should not be nessecary
  // E X P E R I M E N T A L
  const double starty=LLspectrum.getcounts(0);
  const double estart=LLspectrum.getenergy(0);
  const int n=LLspectrum.getnpoints();
  const double endy=LLspectrum.getcounts(n-1);
  const double eend=LLspectrum.getenergy(n-1);
  const double slope=(endy-starty)/(eend-estart);
  const double cliplimit=-20.0;
for (int i=0;i<n;i++){
    const double E=LLspectrum.getenergy(i);
    const double cts=LLspectrum.getcounts(i);
    //E X P E R I M E N T
    const double correction=starty+(E-estart)*slope;
    LLspectrum.setcounts(i,cts-correction);
    //clip off any remaining counts that are too negative
    if (cts-correction<cliplimit){
      LLspectrum.setcounts(i,cliplimit);
    }
  }


   //show this spectrum as a test
   //Spectrum* myspec=new Spectrum(n);
   //copy because LLspectrum will die when out of scope
   //(*myspec)=LLspectrum;
   //myspec->display(0);

  //normalize the spectrum to 1
  LLspectrum.normalize();

   //check if the maximum of the spectrum is close to 0eV energy, otherwise callibration is wrong
  ZLindex=LLspectrum.getmaxindex();
  double e0=LLspectrum.getenergy(ZLindex);
  if (fabs(e0)>10.0){
      //careful, for really thick specimen this fails and we should keep the calibrated
    Saysomething mysay(0,"warning","The position of the zero loss peak is more then 10eV off (should be close to 0eV), keeping calibrated units");
    ZLindex=LLspectrum.getenergyindex(0.0);
  }
  copytobuffer(&LLspectrum, realLL, ZLindex); //copy spectrum into fft buffer, pad zeros and shift

  //perform fft
  fft_planLL->fft(realLL,fourierLL);
  //result is now stored in fourierLL

/*
  //test do backward transform
  ifft_planLL->fftNormalized(fourierLL,realLL);
  for (size_t i=0;i<(this->getnpoints());i++){
    LLspectrum.setcounts(i,realLL[i]);
  }
  LLspectrum.display(getworkspaceptr());
*/
}

   
'''         
            