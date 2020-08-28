# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import matplotlib.pyplot as plt

print(keras.__version__)

class Encoder extends AbstractRNNLayer
{
    protected $backend;
    protected $builder;
    protected $vocabSize;
    protected $wordVectSize;
    protected $recurrentUnits;
    protected $embedding;
    protected $rnn;

    public function __construct(
        $backend,
        $builder,
        string $rnn,
        int $input_length,
        int $vocab_size,
        int $word_vect_size,
        int $recurrent_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->rnnName = $rnn;
        if($rnn=='simple') {
            $this->rnn = $builder->layers()->SimpleRNN(
                $recurrent_units,[
                    'return_state'=>true,
                ]);
        } elseif($rnn=='lstm') {
            $this->rnn = $builder->layers()->LSTM(
                $recurrent_units,[
                    'return_state'=>true,
                ]);
        } else {
            throw new InvalidArgumentException('unknown rnn type: '.$rnn);
        }
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->rnn,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->rnn->statesShapes();
        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'rnn'=>$this->rnnName,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->rnn->forward($wordvect,$training,$initalStates);
        return [$outputs,$states];
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        [$dWordvect,$dStates]=$this->rnn->backward($dOutputs,$dNextStates);
        $dInputs = $this->embedding->backward($dWordvect);
        return [$dInputs,$dStates];
    }
}

class Decoder extends AbstractRNNLayer
{
    protected $backend;
    protected $builder;
    protected $vocabSize;
    protected $wordVectSize;
    protected $recurrentUnits;
    protected $denseUnits;
    protected $embedding;
    protected $rnn;
    protected $dense;

    public function __construct(
        $backend,
        $builder,
        string $rnn,
        int $input_length,
        int $vocab_size,
        int $word_vect_size,
        int $recurrent_units,
        int $dense_units
    )
    {
        $this->backend = $backend;
        $this->inputShape=[$input_length];
        $this->vocabSize = $vocab_size;
        $this->wordVectSize = $word_vect_size;
        $this->recurrentSize = $recurrent_units;
        $this->denseUnits = $dense_units;

        $this->embedding = $builder->layers()->Embedding($vocab_size, $word_vect_size);
        $this->rnnName = $rnn;
        if($rnn=='simple') {
            $this->rnn = $builder->layers()->SimpleRNN(
                $recurrent_units,[
                    'return_state'=>true,
                    'return_sequence'=>true,
                ]);
        } elseif($rnn=='lstm') {
            $this->rnn = $builder->layers()->LSTM(
                $recurrent_units,[
                    'return_state'=>true,
                    'return_sequence'=>true,
                ]);
        } else {
            throw new InvalidArgumentException('unknown rnn type: '.$rnn);
        }
        $this->dense = $builder->layers()->Dense($dense_units);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShape($inputShape);
        $inputShape = $this->registerLayer($this->embedding,$inputShape);
        $inputShape = $this->registerLayer($this->rnn,$inputShape);
        $inputShape = $this->registerLayer($this->dense,$inputShape);
        $this->outputShape = $inputShape;
        $this->statesShapes = $this->rnn->statesShapes();

        return $this->outputShape;
    }

    public function getConfig() : array
    {
        return [
            'builder'=>true,
            'rnn'=>$this->rnnName,
            'vocab_size'=>$this->vocabSize,
            'word_vec_size'=>$this->wordVecSize,
            'recurrent_units'=>$this->recurrentUnits,
            'dense_units'=>$this->denseUnits,
            ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $wordvect = $this->embedding->forward($inputs,$training);
        [$outputs,$states]=$this->rnn->forward($wordvect,$training,$initalStates);
        $outputs=$this->dense->forward($outputs,$training);
        return [$outputs,$states];
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $dOutputs = $this->dense->backward($dOutputs);
        [$dWordvect,$dStates]=$this->rnn->backward($dOutputs);
        $dInputs = $this->embedding->backward($dWordvect);
        return [$dInputs,$dStates];
    }
}

class Seq2seq extends AbstractModel
{
    use GenericUtils;
    protected $encode;
    protected $decode;
    protected $encoutShape;

    public function __construct($backend,$builder,array $options=null)
    {
        extract($this->extractArgs([
            'rnn'=>null,
            'input_length'=>null,
            'input_vocab_size'=>null,
            'target_vocab_size'=>null,
            'word_vect_size'=>8,
            'recurrent_units'=>256,
            'dense_units'=>256,
            'start_voc_id'=>0,
        ],$options));
        parent::__construct($backend,$builder,$builder->utils()->HDA());
        $this->encoder = new Encoder(
            $backend,$builder,
            $rnn,
            $input_length,
            $input_vocab_size,
            $word_vect_size,
            $recurrent_units
        );
        $this->decoder = new Decoder(
            $backend,$builder,
            $rnn,
            $input_length,
            $target_vocab_size,
            $word_vect_size,
            $recurrent_units,
            $dense_units
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->setLastLayer($this->out);
        $this->startVocId = $start_voc_id;
    }

    protected function buildLayers(array $options=null) : void
    {
        $this->registerLayer($this->encoder);
        $shape = $this->registerLayer($this->decoder);
        $this->registerLayer($this->out,$shape);
    }

    protected function shiftSentence(
        NDArray $sentence)
    {
        $K = $this->backend;
        $result = $K->zerosLike($sentence);
        [$batches,$length] = $sentence->shape();
        for($batch=0;$batch<$batches;$batch++){
            $source = $sentence[$batch][[0,$length-2]];
            $dest = $result[$batch][[1,$length-1]];
            $result[$batch][0]=$this->startVocId;
            $K->copy($source,$dest);
        }
        return $result;
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $K = $this->backend;
        [$dummy,$states] = $this->encoder->forward($inputs,$training,null);
        $this->encoutShape = $dummy->shape();
        $dec_inputs = $this->shiftSentence($trues);
        [$outputs,$dummy] = $this->decoder->forward($dec_inputs,$training,$states);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
    }

    protected function backwardStep(NDArray $dout) : NDArray
    {
        $K = $this->backend;
        $dout = $this->out->backward($dout);
        [$dummy,$dStates] = $this->decoder->backward($dout,null);
        [$dInputs,$dStates] = $this->encoder->backward($K->zeros($this->encoutShape),$dStates);
        return $dInputs;
    }

    public function translate(NDArray $sentence)
    {
        $K = $this->backend;
        $inputLength = $sentence->size();
        $sentence = $sentence->reshape([1,$inputLength]);
        $this->setShapeInspection(false);
        [$dmy,$states]=$this->encoder->forward($sentence,$training=false);
        $vocId = $this->startVocId;
        $targetSentence =[];
        for($i=0;$i<$inputLength;$i++){
            $in = $K->array([[$vocId]]);
            [$predictions,$states] = $this->decoder->forward($in,$training=false,$states);
            $vocId = $K->argMax($predictions);
            $targetSentence[]=$vocId;
        }
        $this->setShapeInspection(true);
        return $K->array($targetSentence);
    }
}



class DecHexDataset:

    def __init__(self):
        self.vocab_input = ['@','0','1','2','3','4','5','6','7','8','9',' ']
        self.vocab_target = ['@','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',' ']
        self.dict_input = dict((v,k) for k,v in self.vocab_input.items())
        self.dict_target = dict((v,k) for k,v in self.vocab_target.items())

    def dicts(self):
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(self,corp_size,length):
        sequence = np.zeros([corp_size,length])
        target = np.zeros([corp_size,length])
        numbers = np.random.choice(corp_size,corp_size)
        for i in range(corp_size):
            num = numbers[i];
            dec = str(num);
            hex = hex(num);
            self.str2seq(
                dec,
                self.dict_input,
                sequence[i]);
            self.str2seq(
                hex,
                self.dict_target,
                target[i])
        
        return [sequence,target]
    
    
    
    def str2seq(
        input_string: str,
        word_dic: dict,
        output_seq: ndarray):
        '''translate string to sequence'''
        sseq = list(input_string.upper())
        seq_len = len(sseq)
        sp = word_dic[' ']
        bufsz=output_seq.size
        for i in range(bufsz):
            if i<seq_len:
                output_seq[i] = word_dic[sseq[i]]
            else:
                output_seq[i]=sp

    def seq2str(
        input_seq: ndarray,
        word_dic: dict
        ) -> str:
        '''translate string to sequence'''
        outout_str = ''
        bufsz=input_seq.size
        for i in range(bufsz):
            output_str += word_dic[input_buf[i]]
            
        return output_str

    public function translate($model,$str)
    {
        $inputs = $this->mo->zeros([1,$this->length]);
        $this->str2seq(
            $str,$this->dict_input,$inputs[0]);
        $target = $model->translate($inputs);
        return $this->seq2str(
            $target,$this->vocab_target
            );
    }

    public function loadData($corp_size,$path=null)
    {
        $this->length = strlen(strval($corp_size));
        if($path==null){
            $path='dec2hex-dataset.pkl';
        }
        if(file_exists($path)){
            $pkl = file_get_contents($path);
            $dataset = unserialize($pkl);
        }else{
            $dataset = $this->generate($corp_size,$this->length);
            $pkl = serialize($dataset);
            file_put_contents($path,$pkl);
        }
        return $dataset;
    }

}
$rnn = 'lstm';
$corp_size = 10000;
$test_size = 100;
$mo = new MatrixOperator();
$backend = new Backend($mo);
$nn = new NeuralNetworks($mo,$backend);
$dataset = new DecHexDataset($mo);
[$dec,$hex]=$dataset->loadData($corp_size);
$train_inputs = $dec[[0,$corp_size-$test_size-1]];
$train_target = $hex[[0,$corp_size-$test_size-1]];
$test_input = $dec[[$corp_size-$test_size,$corp_size-1]];
$test_target = $hex[[$corp_size-$test_size,$corp_size-1]];
$input_length = $train_inputs->shape()[1];
[$iv,$tv,$input_dic,$target_dic]=$dataset->dicts();
$input_vocab_size = count($input_dic);
$target_vocab_size = count($target_dic);

echo "[".$dataset->seq2str($train_inputs[0],$dataset->vocab_input)."]=>[".$dataset->seq2str($train_target[0],$dataset->vocab_target)."]\n";

$seq2seq = new Seq2seq($backend,$nn,[
    'rnn'=>$rnn,
    'input_length'=>$input_length,
    'input_vocab_size'=>$input_vocab_size,
    'target_vocab_size'=>$target_vocab_size,
    'start_voc_id'=>$dataset->dict_target['@'],
    'word_vect_size'=>16,
    'recurrent_units'=>512,
    'dense_units'=>512,
]);

$seq2seq->compile([
    'optimizer'=>$nn->optimizers()->Adam(),
    ]);
$history = $seq2seq->fit($train_inputs,$train_target,
    ['epochs'=>5,'batch_size'=>128,'validation_data'=>[$test_input,$test_target]]);

$samples = ['10','255','1024'];
foreach ($samples as $value) {
    $target = $dataset->translate(
        $seq2seq,$value);
    echo "[$value]=>[$target]\n";
}
