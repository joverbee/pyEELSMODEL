from pyEELSMODEL.core.component import Component
from pyEELSMODEL.core.parameter import Parameter
from pyEELSMODEL.core.multispectrum import MultiSpectrum
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
import numpy as np
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)


class GDOSLin(Component):
    """
    Linear fine structure DOS Component
    This component will create in the onset region [estart-estart+ewidth] of a core loss a 
    interpolated curve going through degree points which are free parameters of the model
    
    This curve is then added to the core loss cross section and will increase or decrease
    certain areas in the cross section
    
    As this is added, this is also a linear component

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model

    estart: float [eV]
        The onset value of the fine structure. This is mostly
        the onset energy of the edge on wich you want to apply the
        fine structure.

    ewidth : float [eV]
        The energy width over which the fine structure is modelled.
        (default: 50)

    degree: int
        The number of parameters used to model the fine structure.
        (default: 20)

    interpolationtype: string
        The type of interpolation used.

    Returns
    -------

    """

    def __init__(self, specshape,estart,ewidth=50,degree=20,interpolationtype='linear'):
        super().__init__(specshape)
        self.setdescription("Linear Fine Structure used to alter atomic cross sections in the near edge region")
        
        
        self.degree=degree
        self.interpolationtype=interpolationtype
  
        p0 = Parameter("Estart",estart,False)
        self._addparameter(p0)
        
        p1 = Parameter("Ewidth",ewidth,False)
        self._addparameter(p1)
        
        
        #and a list of variables controlling the shape of the fine struct
        for i in range(self.degree):
            pname='a'+str(i)
            p=Parameter(pname,1.0,True)
            p.setboundaries(-np.Inf, np.Inf)
            p.setlinear(True) #is this true as we will multiply this with another cross section  
            self._addparameter(p)

        self.ctes = np.ones(self.degree+2)
        self.initDOS()

        self.connected_edge = None
        self.set_gdos_name()
        self.calculate_integral_per_parameter()


    @classmethod
    def gdoslin_from_edge(cls, specshape, component, pre_e=5, ewidth=50, degree=20,
                          interpolationtype='linear'):
        """
        Class method is made to create an gdoslin which is connected to a coreloss edge.
        No need to input the onset energy

    Parameters
    ----------
    specshape : Spectrumshape
        The spectrum shape used to model

    component: CoreLossEdge
        The coreloss edge on which to add the fine structure.

    pre_e: float [eV]
        The value of the energy onset of the fine structure with respect
        to the onset energy of the coreloss edge. Hence for oxygen at 532 eV
        with pre_e=5, the starting energy for the fine structure is 527 eV.
        (default: 5)

    ewidth : float [eV]
        The energy width over which the fine structure is modelled.
        (default: 50)

    degree: int
        The number of parameters used to model the fine structure.
        (default: 20)

    interpolationtype: string
        The type of interpolation used.

    Returns
    -------

        """


        if not isinstance(component, CoreLossEdge):
            raise TypeError(r'Component should be a CoreLossEdge')

        estart = component.onset_energy - pre_e
        comp = GDOSLin(specshape,estart,ewidth=ewidth,degree=degree,
                       interpolationtype=interpolationtype)
        comp.connected_edge = component
        # print(comp.connected_edge)
        comp.set_gdos_name()
        return comp




    def set_gdos_name(self):
        """

        """
        s1 = "Linear Fine Structure (DOS):"
        s2 = 'Onset' +str(self.parameters[0].getvalue())+'eV'
        s3 = 'Width' + str(self.parameters[1].getvalue()) + 'eV: '
        if self.connected_edge is None:
            self._setname(' '.join([s1, s2, s3]))
        else:
            self._setname(' '.join([s1, s2, s3, self.connected_edge.name[:2]]))

    def calculate(self):  
        p0=self.parameters[0]
        p1=self.parameters[1]
        Estart= p0.getvalue()
        Ewidth= p1.getvalue()
        Estop=Estart+Ewidth
        

        #calculate the DOS
        changes=False
        
        for i in range(self.degree):
            p=self.parameters[i+2]
            changes=changes or p.ischanged()

        if p0.ischanged() or changes or p1.ischanged():
            logger.debug("parameters changed calculating DOS. degree: %d", self.degree)
            logger.debug("Estart: %e", Estart)
            logger.debug("Estop: %e", Estop)
            
            en=self.energy_axis
                            
            self.initDOS()

            yp = self.yp / self.ctes
            # yp = self.yp

            f = interp1d(self.Ep, yp, kind=self.interpolationtype)
            self.data[self.indexstart:self.indexstop]=f(en[self.indexstart:self.indexstop])

            #set parameters as unchanged since last time we calculated
            self.setunchanged()
  
        else:
          logger.debug("parameters have not changed, i don't need to calculate again")

    def initDOS(self):
        #prepare all for the DOS to work with the given options
        p0=self.parameters[0]
        p1=self.parameters[1]
        Estart= p0.getvalue()
        Ewidth= p1.getvalue()
        Estop=Estart+Ewidth  
        en=self.energy_axis
        self.indexstart= self.get_energy_index(Estart)
        if self.get_energy_index(Estop) == self.size:
            self.indexstop= self.get_energy_index(Estop)-1
        else:
            self.indexstop = self.get_energy_index(Estop)
        if self.indexstop<=self.indexstart:
            logger.warning('Estart and Estop are too close together')
            return

        self.Ep=[]
        self.yp=[]
        Estep=(en[self.indexstop]-en[self.indexstart])/(self.degree+1)
        #start with yp=0 at estart
        self.Ep.append(en[self.indexstart])
        self.yp.append(0)
        for i in range(self.degree):
            self.Ep.append(en[self.indexstart]+(i+1)*Estep)
            self.yp.append(self.parameters[i+2].getvalue())
        self.Ep.append(en[self.indexstop])
        self.yp.append(0)
        self.yp = np.array(self.yp)

    def couple_parameter_to_edge(self):
        """
        Couples the parameters of the gdoslin to the edge such that this template
        edge can be used to fit the spectrum where the gdoslin is no variable anymore.
        #todo still needs to be tested
        """
        if self.connected_edge is None:
            print('no edge is connected to this gdoslin')
            return

        for i in range(self.degree):
            p = self.parameters[i + 2]
            fraction = p.getvalue()/self.connected_edge.parameters[0].getvalue()
            p.couple(self.connected_edge.parameters[0], fraction=fraction)


    def calculate_integral_per_parameter(self):
        """
        Since the integral for each individual parameter is not the same we
        calculate the difference which is then applied to the component
        as a multiplication factor to resolve this interpolation problem
        """
        ctes = np.ones(len(self.parameters))

        for ii, param in enumerate(self.parameters[2:]):
            for par in self.parameters[2:]:
                if par == param:
                    par.setvalue(1)
                else:
                    par.setvalue(0)
            self.calculate()

            # ctes[ii+1] = self
            ctes[ii+1] = self.data.sum()
            ctes[ii+1] /= (self.Ep[ii+2] - self.Ep[ii]) / (self.Ep[2] - self.Ep[0])

        self.ctes = ctes
        # print('done')

        for param in self.parameters[2:]:
            param.setvalue(1)

        self.calculate()



'''



Spectrum* GDoslin::getgradient(size_t j){
  //get analytical partial derivative to parameter j and return pointer to spectrum containing the gradient
	
	
	//TODO check gradients, they seem to cause a memory allocation problem in the fitter
	


  //get the parameters
  const Parameter* Estartptr= getparameter(0);
  const double Estart=Estartptr->getvalue();
  const Parameter* Ewidthptr= getparameter(1);
  const double Estop=Estart+Ewidthptr->getvalue();

  int jindex=(int)j-2;
  if ((jindex<0)||(jindex>=(int)degree)){
    //don`t have a derivative for these
    throw Componenterr::bad_index();
    return &gradient;
  }

  const double edistance=(Estop-Estart)/double(degree+1);
  if (edistance<1e-3){
      Saysomething mysay(0,"Error","Estart must be < Estop, not calculating gradient");
      return &gradient;
  }

  //get energy of the parameter and energy of previous and next parameter
  const double  ej=Estart+double(jindex+1)*edistance;
  //get previous
  const double eprevious=Estart+double(jindex)*edistance;
  //get next
  const double enext=Estart+double(jindex+2)*edistance;

  //analytical derivative wrt parameter j
  for (unsigned int i=0;i<this->getnpoints();i++)
  {
      double en=this->getenergy(i);
      //if en>previous and < next we have a simple deriv
      //otherwise zero
      try{
	if ((en>eprevious)&&(en<enext)&&(en<Estop)&&(en>Estart)){
	  if (en<=ej){
        gradient.setcounts(i,((en-eprevious)/edistance));
	  }
	  else{
        gradient.setcounts(i,(1.0-(en-ej)/edistance) );
	  }
	}
	else{
	  gradient.setcounts(i,0.0);
	}
      }
      catch(...){
	throw Componenterr::bad_index();
      }
  }

  return &gradient;
}

GDoslin* GDoslin::clone()const{
  return new GDoslin(*this);
}
GDoslin* GDoslin::new_component(int n,double estart,double dispersion,std::vector<Parameter*>* parameterlistptr,Model * m)const{
  return new GDoslin(n,estart,dispersion,parameterlistptr,m);
}

void GDoslin::rescaleavg(){
    //scale parameters to avg, this is handy if you do the calibration to the sum rule
    //since otherwise the scaling has no true meaning and the parameters start to drift off towards the boundaries
Parameter* pointptr=0;
    //determine avg
    double avg=0.0;
    for (size_t index=0;index<degree;index++){
    pointptr=getparameter(index+3);
    avg+=pointptr->getvalue();
   }
   avg=avg/degree;
   //apply it to the points
   for (size_t index=0;index<degree;index++){
    pointptr=getparameter(index+3);
    const double newval=pointptr->getvalue()/avg;
    pointptr->setvalue(newval);
   }

}
void GDoslin::preparesubspace(){
    //create a subsampled space, do this whenever Estart or Estop changes
    //we could easily cope with changing number of parameters
    //but the only question is how to do this in the user interface???

  //total number of points in this space
    const Parameter* Estartptr= getparameter(0);
    const double Estart=Estartptr->getvalue();
    const Parameter* Ewidthptr= getparameter(1);
    const double Estop=Estart+Ewidthptr->getvalue();
    subdispersion=(fabs(Estart-Estop))/double(degree);
    nsubspace=size_t(fabs(this->getenergy(0)-this->getenergy(this->getnpoints()-1))/subdispersion);
    pointsafter=size_t((this->getenergy(this->getnpoints()-1)-Estop)/subdispersion);
    pointsbefore=size_t((Estart-this->getenergy(0))/subdispersion);
    if (pointsafter>nsubspace){
        pointsafter=nsubspace;
    }
   


    #ifdef COMPONENT_DEBUG
    std::cout << "DOS creating subspace subdispersion: " << subdispersion <<"\n";
    std::cout << "pointsbefore: " << pointsbefore <<"\n";
    std::cout << "pointsafter: " << pointsafter <<"\n";
    std::cout << "nsubspace: " << nsubspace <<"\n";
    #endif

    //and prepare the fourier stuff

    if (realSUB!=0){
        FFTWdelete(realSUB);
        FFTWdelete(fourierSUB);
    }

    realSUB = FFTWdouble(2*nsubspace);
    fourierSUB =FFTWComplex(nsubspace+1);

    //make a plan for Fourier transforms
    fft_planSUB=new rcfft1d(2*nsubspace,realSUB,fourierSUB);
    ifft_planSUB=new crfft1d(2*nsubspace,fourierSUB,realSUB);

}

void GDoslin::setoptions(){
    //call a function to set the options of the DOS
    std::string optionstring=interactiveoptions();
    storeoptions(optionstring);
    initDOS();
    //force an update of the model
    Parameter* firstptr= getparameter(3);
    firstptr->forcechanged(); //otherwise the model doesn't want to calculate
    geteelsmodelptr()->componentmaintenance_doupdate(); //redraw the parameters in componentmaintenance
    geteelsmodelptr()->componentmaintenance_updatescreen();
    geteelsmodelptr()->slot_componentmaintenance_updatemonitors();//make sure the monitors are recalculated
}

void GDoslin::initDOS(){
    //prepare all for the DOS to work with the given options

    //create an extra plot for sum rule work
    if (cumsumrule!=0){
        delete(cumsumrule);
        cumsumrule=0;
    }
    if (dosumrule){
        cumsumrule=new Spectrum(this->getnpoints(),this->getenergy(0),this->getdispersion());
        cumsumrule->setname("cumulative Bethe sum rule difference");
        cumsumrule->display(getworkspaceptr());
    }

    InitEnergy(); //prepare the energy points that are linked to the parameters
    //create a special plot
    if (this->isvisible()){
        if (Plotspec!=0){
            //remove it from the plot first
            this->getgraphptr()->removelastgraph();
            Plotspec->resize(Evector.size());
        }
        else{
            Plotspec=new Spectrum(Evector.size());
        }
    }
   
    initplotspec();
    if (this->isvisible()){
        (this->getgraphptr())->addgraph(Plotspec);
        (this->getgraphptr())->setstyle(1,2); //set style of this plot to dots instead of lines
    }
    switch(interpolationtype){
        case 1:
            break;
        case 2:
            //cubic spline type
            gslinit(); //setup the memory for the gsl fitter
            break;
        case 3:
            //upsample type
            preparesubspace();
            break;
        default:
            //unknown, go to linear
            interpolationtype=1;
        }

    //we only have analytical gradients in case of linear energy sampling
    //tell that we have gradients for the GDoslin points
    //parameters 4....degree+4
    for (size_t i=4;i<(degree+4);i++){
    	//this->sethasgradient(i,(broadeningtype==CONSTANT)&&(interpolationtype==1));
    	//gradients don't seem to work, they cause a memory alocation problem????
    	this->sethasgradient(i,false);
    }    
}

std::string GDoslin::interactiveoptions(){
     GDosoptions* myoptions=new GDosoptions(getworkspaceptr(),"",degree,interpolationtype,dosumrule,(size_t)broadeningtype,smoothvalue,threshold);
        const int result=myoptions->exec();
        std::string optionstring="";
        if (result==1){
            //OK pressed
            size_t olddegree=degree;
            degree=myoptions->getdegree();
            dosumrule=myoptions->getsumrule();
            interpolationtype=myoptions->getinterpolation();
            smoothvalue=myoptions->getintegrationwidth();
            threshold=myoptions->getthreshold();
            broadeningtype=(broadening)myoptions->getbroadeningtype();
            if (interpolationtype==3){
                broadeningtype=CONSTANT;
                optionstring="Constant broadening";
            }

            //do consistency check
            if ((degree<4)||(degree>this->getnpoints())){
                degree=4;
            }

            if (olddegree!=degree){
                //degree changed, change number of parameters and put all parameters to 1
                //but only if this component already exists
                if (this->get_nr_of_parameters()>4){
                    changedegree(olddegree,degree);
                }
            }
            if ((threshold<0.0)||(threshold>1.0)){
                threshold=0.5;
            }
        }
       //create an optionstring
        switch(broadeningtype){
                case LINEAR:
                    optionstring="Linear coefficient";
                    break;
                case QUADRATIC:
                    optionstring="Quadratic coefficient";
                    break;
                case EGERTON:
                    optionstring="Egerton Broadening atomic distance [nm]";
                    break;
                default:
                    optionstring="Constant broadening";
                    broadeningtype=CONSTANT;
                    break;
        }

        return optionstring;
     }
double GDoslin::getoptions(){
     //create a coded options code to store in a parameter
     double options=0;
     if (dosumrule){
         options=double(interpolationtype)+threshold;
    }else{
        options=-double(interpolationtype)-threshold;
    }
return options;
}
void GDoslin::storeoptions(std::string optionstring){
            Parameter* optionptr= getparameter(2);
            optionptr->setchangeable(true);
            optionptr->setvalue(getoptions());
            optionptr->setname(optionstring);
            optionptr->setchangeable(false);
}
void GDoslin::makeoptions(){
    Parameter* optionptr= getparameter(2);
    double options=optionptr->getvalue();
    threshold=fabs(options)-std::floor(fabs(options));   //treshold stored as behind the comma between 0 and 1
    if ((threshold<=0.0)||(threshold>1.0)){
        threshold=0.25;
    }
    if (options<0.0){
        dosumrule=false;
        interpolationtype=size_t(std::floor(fabs(options)));
    }
    else{
        dosumrule=true;
        interpolationtype=size_t(std::floor(fabs(options)));
    }
    //consistency check
    if ((interpolationtype==0)||(interpolationtype>3)){
        //unknown type, reset to piecewise linear
        interpolationtype=1;
    }
}

 void GDoslin::changedegree(size_t olddegree,size_t degree){
     //change number of degrees
     if (degree>olddegree){
         //add new points at end if olddegree<degree
         for (size_t i=0;i<(degree-olddegree);i++){
            std::string name;
            std::ostringstream s;
            if ((s << "a"<< i+olddegree)){ //converting an int to a string in c++ style rather than unsafe c-style
                name=s.str();
            }
            Parameter* p=new Parameter(name,1.0,1);
            p->setboundaries(-1.0e5,1.0e5);
            p->setlinear(true); //these parameters are linear
            this->addparameter(p);
        }
    }
    else{
        //else remove them
        for (size_t i=0;i<(olddegree-degree);i++){
            this->pullparameter();
        }
    }
    //tell the model that something changed
    //this kills the stored values however
    Model* mymodel=this->getmodelptr();
    mymodel->resetstorage();
   // this->showequalizer(); //redraw the equaliser with new number of sliders
}
double  GDoslin::Lifetimebroadening(double E){
  //calculate Lifetime broadening in eV according to Egerton 2007
  //E is the energy in eV above the onset

    const double dEmin=this->getdispersion(); //the minimum energy step is the dispersion
    const double dEmax=100.0; //maximum lifetime broadening, more doesn't make sense
    const double epsilon=E;
    const double h=4.13566733e-15; //eV/s
    const double m0=9.10938188e-31; //electron mass in kg
    const double e=1.60217646e-19; //electron charge (C)
    const double m=m0;
    const Parameter* aptr=getparameter(3);
    const Parameter* optptr=getparameter(2);
    const double a=aptr->getvalue(); //atomic diameter in nm
    double lambda=0.0;
    double v=0.1;

    double tau=0.0;
    const double pi=acos(-1.0);
    const Parameter* Estartptr= getparameter(0);
    const double Estart=Estartptr->getvalue();
    const Parameter* Ewidthptr= getparameter(1);
    const double Estop=Estart+Ewidthptr->getvalue();
    //#ifdef COMPONENT_DEBUG
    //  std::cout <<"Z="<<Z<<" rho="<<rho<<" \n";
    //#endif
    broadeningtype=CONSTANT;
    if (optptr->getname()=="Linear coefficient") broadeningtype=LINEAR;
    if (optptr->getname()=="Quadratic coefficient") broadeningtype=QUADRATIC;
    if (optptr->getname()=="Egerton Broadening atomic distance [nm]") broadeningtype=EGERTON;
    if (a==0.0) broadeningtype=CONSTANT;
    double dE;

switch(broadeningtype)
{
    case EGERTON:
        //Egerton broadening
        if (epsilon>this->getdispersion()){
            v=sqrt(2.0*e*epsilon/m); //speed
        }
        lambda=538.0*fabs(a)*pow(fabs(epsilon),-2.0)+0.41*pow(fabs(a),3.0/2.0)*sqrt(fabs(epsilon));
        tau=lambda*1e-9/v; //lifetime in seconds
        //#ifdef COMPONENT_DEBUG
        //std::cout <<"v="<<v<<" tau="<<tau<<" lambda"<<lambda<<" a="<<a<<" \n";
        //#endif

        dE=h/(2.0*pi*tau);
        //if dE lower than dispersion or E< Eonset
       break;
    case QUADRATIC:
        //Quadratic broadening
        dE=fabs(a)*pow(fabs(epsilon),2.0); // a simple quadratic broadening
        //check for infinity
        break;
    case LINEAR:
        //broadening linear with energy above onset
         dE=fabs(a)*fabs(epsilon); // a simple linear broadening
         break;
    case CONSTANT:
    default:
        dE=(Estart-Estop)/degree;
        break;
}

if (dE<dEmin){
    dE=dEmin;
}
if (E<=0){
    dE=dEmin;
}
if (dE>dEmax){
    dE=dEmax;
}

//#ifdef COMPONENT_DEBUG
//std::cout <<"Returning a lifetime broadening of dE="<<dE<<"for energy E="<<E<<"\n";
//#endif
return dE;
}
void GDoslin::InitEnergy(){
#ifdef COMPONENT_DEBUG
    std::cout <<"Initialising Energy points\n";
#endif
  //do this at the start and whenever the Estart or Estop changes
  size_t Eminid=0;
  size_t Emaxid=0;
  const Parameter* Estartptr=this->getparameter(0);
  const double Estart=Estartptr->getvalue();
  Parameter* Ewidthptr= this->getparameter(1);
  double Estop=Estart+Ewidthptr->getvalue();

  for (size_t bin=0;bin<this->getnpoints();bin++){
    if (this->getenergy(bin)<=Estart){
      Eminid=bin;
    }
    if (this->getenergy(bin)<=Estop){
      Emaxid=bin;
    }
  }
#ifdef COMPONENT_DEBUG
  std::cout <<"Eminid: "<<Eminid<<"\n";
  std::cout <<"Emaxid: "<<Emaxid<<"\n";
#endif
    // copy parameters in Yvector and initialise
    Yvector.clear();
    Evector.clear();

    //add a first point that is zero and has energy Estart
    //if offset=2 add another point to make sure the aproach is flat
    if (offset==2){
        Yvector.push_back(0.0);
        Evector.push_back(Estart-1.0);
    }
    Yvector.push_back(0.0);
    Evector.push_back(Estart);
    //add points defined by the parameters
    //determine where we end up in terms of E when doing the Energy sum
    double Eposp=Estart;
    for (size_t index=0;index<=degree;index++){//do one more to end up on the point that should be Estop
        Eposp=Eposp+Lifetimebroadening(Eposp-Estart);
    }
    //Eposp is now the energy where we would end if taking the broadening
    //we want the energy to end at Estop
    //so we rescale with
    const double scale=(Eposp-Estart)/(Estop-Estart);
    double Epos=Estart;
    for (size_t index=0;index<degree;index++){
        //copy the parameter values in a vector
        Yvector.push_back((this->getparameter(index+4))->getvalue());
        Epos=Epos+Lifetimebroadening(Epos-Estart);
        if (interpolationtype!=3){
            Evector.push_back(Estart+(Epos-Estart)/scale);
        }
        else{
            Evector.push_back(Estart+((double)(index)+0.5)*(Estop-Estart)/(double)(degree));
        }

    }

    //recalculate Estop

    //add a last point that is 1 at Estop
    Yvector.push_back(0.0);
    Evector.push_back(Estop);
    //a second last point to make sure the end of the spline aproaches the tail with a flat section
    if (offset==2){
        Yvector.push_back(0.0);
        Evector.push_back(Estop+1.0);
    }

#ifdef COMPONENT_DEBUG
std::cout <<"Evector\n";
for (size_t i=0;i<Evector.size();i++){
  std::cout <<"Evector[ "<<i<<"]="<<Evector[i]<<"\n";
}
#endif

}

void GDoslin::copyparameters(){
    //copy the paramters values in the Yvector
    for (size_t index=0;index<degree;index++){
        //copy the parameter values in a vector, remember that point Yvector[0] is the first point but not connected to the parameters
        Yvector[index+offset]=((this->getparameter(index+4))->getvalue());
    }
}
void GDoslin::dospline(){
    //calculate the spline coefficients
     gsl_spline_init (sp, &Evector[0], &Yvector[0], Evector.size());
}

double GDoslin::seval(double x){
    return gsl_spline_eval (sp, x, acc);
}


void GDoslin::gslinit(){
    //init a cubic spline
    acc= gsl_interp_accel_alloc ();
    sp= gsl_spline_alloc (gsl_interp_cspline, Evector.size());
}
void GDoslin::initplotspec(){
    //copy Yvector and Evector in plotspec
    if (Plotspec!=0){
        for (size_t i=0;i<Plotspec->getnpoints();i++){
            Plotspec->setdatapoint(i,Evector[i],Yvector[i],0.0);
        }

        if (this->isvisible()) {
            this->getgraphptr()->updategraph(1,Plotspec);
        }

    }
}

//inherit show but also redefine what should happen with the dos window
void GDoslin::show(){
    Component::show(); //do the normal show
    if (Plotspec!=0){
        initplotspec(); //but also update the points on the plot

        //copy energy values in the names of the parameters
        //this gives users the possiblity to know at what energy the dos points where taken
        for (size_t index=0;index<degree;index++){
            Parameter* pointptr=getparameter(index+4);
            std::string namestring;
            std::ostringstream s;
            if (s << "a" << index <<" @ " << Evector[index+offset] <<" eV"){ //converting an int to a string in c++ style rather than unsafe c-style
                // conversion worked
                namestring=s.str();
            }
            pointptr->setname(namestring);
        }
    }
}






'''
