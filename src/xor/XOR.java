/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package xor;
import java.lang.Math;
import java.util.Random;
/**
 *
 * @author AVELL
 */
public class XOR {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        double [][] x = {{1,1,1},{0,0,1},{1,0,1},{0,1,1}}; // inclui o termo bias na matriz de amostras
        double [] s = {0,0,1,1};
        double w,eta;
        int hl,ol;
        double [][] w1;
        double [] w2;
        double [] ih;
        double [] out_ho;
        double [] io,out;
        double [] dels,delh;
        double fake_s;
        double [][] hidden,fhidden;
        double[] last,res;
        
        Random rand = new Random();
        
        hl = 2; // 2 neuronios na camada intermediaria
        ol = 1; // 1 neuronios na camada de saída

        w1 = new double [3][2]; //3 elementos pq uma das colunas deve ser o peso sináptico do termo bias
        w2 = new double [3]; //3 elementos pq uma das linhas deve ser o peso sináptico do termo bias
        
        for (int i=0;i<w2.length;i++){
            w = rand.nextDouble();
            w2[i] = w;
            System.out.println("w2 " + w2[i]+"\t");
        }
        
        w=0;
        
        for(int i=0;i<w1.length;i++) {       
            for (int j=0;j<w1[0].length;j++) {
                w = rand.nextDouble();
                //w=w+1;
                w1[i][j] = w;
                System.out.print("w1 " + w1[i][j]+" ");
            }
            System.out.print(" \n\n");
        }

        double[] erro;
        
        eta=.7;
        
        
        for (int a=1;a<10000;a++){
            

            for (int i=0;i<x.length;i++){ /*para cada uma das amostras

                Lembrando que a amostra é uma matriz composta pelos vetores
                de entrada da rede neural
                
                */
                
                ih = new double [hl];
                out_ho = new double[hl];
                
                /*
                Input aos neurônios da camada intermediária
                Saída dos neurônios da camada intermediária
                */
                for (int l=0;l<x[i].length-1;l++){ 
                    for (int c=0;c<w1.length-1;c++){ 
                        ih[l]+=x[i][c]*w1[c][l]; //entrada sem considerar o bias
                        }
                    ih[l]+=w1[w1.length-1][l]; //soma do termo bias
                    
                    //out_ho[l] = 1/(1+Math.exp(-ih[l])); //saída
                    
                    out_ho[l] = Math.tanh(-ih[l]); //saida do neuronio da camada intermediaria
                    }
                    
                /*
                Input aos neurônios da camada de saída
                Saída dos neurônios da camada de saída
                */
                
                    io = new double [out_ho.length-1];
                    out = new double[ol];
                    dels = new double[ol];
                
                    for (int l=0;l<out_ho.length-1;l++){
                        for (int c=0;c<out_ho.length;c++){
                            io[l] += out_ho[c]*w2[c]; //Entrada ao neuronio da camada de saida
                        }
                            io[l]+=w2[w2.length-1]; //somar termo bias
                            
                            out[l] = 1/(1+Math.exp(-io[l])); //Saída --> resultado da rede
                            
                            dels[l]=out[l]*(1-out[l])*(s[i]-out[l]); //gradiente local induzido
                        }

                    
                    //fake_s=0;
                    delh = new double[hl];
                    
                    for (int nh=0;nh<hl;nh++){
                        fake_s=0;
                        for (int no=0;no<ol;no++){
                            /*
                            Somatório da multiplicação dos gradientes locais induzidos
                            pelos pesos entre camada intermediária e de saída
                            */
                            fake_s+=w2[nh]*dels[no];
                        }
                        /*
                        Cálculo do gradiente local induzido entre camada intermediária
                        e camada de entrada
                        */
                        //delh[nh]=out_ho[nh]*(1-(out_ho[nh]*out_ho[nh]))*fake_s;
                        delh[nh]=out_ho[nh]*(1-(out_ho[nh]*out_ho[nh]))*fake_s;
                        //delh[nh]=out_ho[nh]*(1-(out_ho[nh]))*fake_s;
                    }
                    
                    for (int no=0;no<ol;no++){
                        for (int nh=0;nh<hl;nh++){
                            w2[nh]+=eta*dels[no]*out_ho[nh];
                        }
                        w2[w2.length-1]+=eta*dels[no]*1; //ajuste do bias                    
                    }
                    
                    for (int nh=0;nh<hl;nh++){
                        for(int ni=0;ni<x[i].length-1;ni++){
                            w1[nh][ni]+=eta*delh[nh]*x[i][ni];
                        }
                        w1[w1.length-1][nh]+=eta*delh[nh]*1; //ajuste do bias
                    }
       
            }
            
        }
            
            hidden = new double [x.length][w1[0].length];
            
            for (int i=0;i<x.length;i++){
                for (int j=0;j<w1[0].length;j++){
                    for (int k=0;k<x[0].length;k++){
                        hidden[i][j]+=x[i][k]*w1[k][j];
                    }
                }
            }
            
            fhidden = new double [hidden.length][hidden[0].length+1];
            
            for (int i=0;i<hidden.length;i++){
                for (int j=0;j<hidden[0].length;j++){
                    //fhidden[i][j]=1/(1+Math.exp(-hidden[i][j]));
                    fhidden[i][j]=Math.tanh(-hidden[i][j]);
                }
                fhidden[i][hidden[0].length]=1;
            }
            
            last = new double [hidden.length];
            
            for (int i=0; i<fhidden.length;i++){
                for (int c=0;c<fhidden[0].length;c++){
                    last[i]+=fhidden[i][c]*w2[c];
                }
            }
            
            res = new double [last.length];
            
            for (int i=0;i<last.length;i++){
                res[i]=1/(1+Math.exp(-last[i]));
                //res[i]=Math.signum(last[i]);
            }
        
            erro = new double [last.length];
            
            for(int i=0;i<last.length;i++){
                erro[i]=Math.pow(s[i]-res[i],2);
            }         
        
            double eqm=0;
            
            for (int i=0;i<res.length;i++){
                eqm+=Math.pow(erro[i],2);
            }
            
            eqm=eqm/(x.length);
            
            System.out.println("-----Resultados & Erros-----");
        for (int i=0;i<res.length;i++){
            System.out.println(res[i]+"\t"+erro[i]+"\t"+eqm);
        }
        
        
    }
    
}
