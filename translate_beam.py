import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument("--data-prefix",
                        help="Prefix attached to file name to marke bpe",
                        type=str,
                        default=""
                        )
    parser.add_argument('--data', default='assignments/03/prepared', help='path to data directory')
    parser.add_argument('--dicts', required=True, help='path to directory containing source and target dictionaries')
    parser.add_argument('--checkpoint-path', default='checkpoints_asg4/checkpoint_best.pt', help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--short-test', action='store_true', help='run short version of test set')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')
    # Add beam search arguments
    parser.add_argument('--nbest', default=0, type=int, help='number of translations created for each reference')
    parser.add_argument('--nbest-ranked', action='store_true', help='use special ranking technique')
    parser.add_argument('--beam-size', default=11, type=int, help='number of hypotheses expanded in beam search')
    # alpha hyperparameter for length normalization (described as lp in https://arxiv.org/pdf/1609.08144.pdf equation 14)
    parser.add_argument('--alpha', default=0.0, type=float, help='alpha for softer length normalization')
    parser.add_argument('--gamma', default=1.0, type=float, help='ranking factor for nbest beamsearch')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.dicts, '{}dict.{:s}'.format(args.data_prefix, args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.dicts, '{}dict.{:s}'.format(args.data_prefix,args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(
            args.data,
            '{}test{}.{:s}'.format(args.data_prefix, '_short' if args.short_test else '', args.source_lang)
        ),
        tgt_file=os.path.join(
            args.data,
            '{}test{}.{:s}'.format(args.data_prefix, '_short' if args.short_test else '', args.target_lang)
        ),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                              batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                         args.batch_size, 1, 0, shuffle=False,
                                                                         seed=args.seed))
    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)

    # Iterate over the test set
    all_hyps = {}
    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch
        batch_size = sample['src_tokens'].shape[0]
        searches = [BeamSearch(args.beam_size, args.max_len - 1, tgt_dict.unk_idx) for i in range(batch_size)]

        with torch.no_grad():
            # Compute the encoder output
            encoder_out = model.encoder(sample['src_tokens'], sample['src_lengths'])
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            # go_slice tensor filled with id of EOS in target language with dimensions batch_sizex1.
            # It's fed to the forward function of the decoder model as a target vector to generate the decoder output states at the first time step (for the batch).
            # Furthermore, a part of the slice is passed on to the SearchBeamNod as part of the sequence.
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])
            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            #import pdb;pdb.set_trace()
            
            # Compute the decoder output at the first time step
            decoder_out, _ = model.decoder(go_slice, encoder_out)

            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            # I think because with the original model it could happen that the last word before EOS is an unknown token.
            # The provided beam search algorithm would in such a case use the back off candidate and its probability.
            # We set the number of candidates to beam size + 1 to assure that also for the last token a backoff exists.
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size+1, dim=-1)

        # Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j+1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j+1]
                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:,i,:]
                lstm_out = encoder_out['src_out'][0][:,i,:]
                final_hidden = encoder_out['src_out'][1][:,i,:]
                final_cell = encoder_out['src_out'][2][:,i,:]
                try:
                    mask = encoder_out['src_mask'][i,:]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, torch.cat((go_slice[i], next_word)), log_p, 1)
                # __QUESTION 3: Why do we add the node with a negative score?
                # See also Question 5. BeamSearch.node is a PriorityQueque. When retrieving from such a queque the smallest itmes are retrieved first.
                # By adding the node with the negative log probability as score we ensure that later on that the most probably node is retrieved.
                searches[i].add(-node.eval(args.alpha), node)

        #import pdb;pdb.set_trace()
        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len-1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]
            if nodes == []:
                break # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size+1, dim=-1)

            # Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                all_nodes_this_sent = []
                for j in range(args.beam_size):
                    # implement special score calculation for ranking here
                    # TODO: use j+1 as the ranking faktor k' from the text
                    # put new node back to nodes and use updated nodes for ranking?
                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j+1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j+1]
                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different? 
                    # What would happen if we did not make this distinction?
                    # add_final assures that all beams have the same length which is required if we want to apply matrix caluclation for batch processing.
                    # Besides that, add_final adds the nod to the final PriorityQueque as we need to keep track of the final nodes in particular (e.g. for masking).

                    # Store the node as final if EOS is generated
                    if next_word[-1] == tgt_dict.eos_idx:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                            next_word)), node.logp, node.length
                            )
                        search.add_final(-node.eval(args.alpha), node)

                    else:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                   next_word)), node.logp + log_p, node.length + 1
                        )
                        all_nodes_this_sent.append((-node.eval(), node))
                # CHANGES
                if args.nbest_ranked:
                    # adjust score with rank
                    sorted_nodes = sorted(all_nodes_this_sent, key=lambda n: n[0], reverse=True)
                    nodes_to_add = []
                    for c, node_tuple in enumerate(sorted_nodes, 1):
                        node_value, node = node_tuple
                        node_value = node_value - c * args.gamma
                        new_node = BeamSearchNode(
                            node.search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, node.sequence, node_value * (-1), node.length
                        )
                        nodes_to_add.append((node_value, new_node))
                    nodes_to_add.sort(key=lambda n: n[0], reverse=True)
                    all_nodes_this_sent = nodes_to_add
                for value, node in all_nodes_this_sent:
                    # Add the node to current nodes for next iteration
                    search.add(value, node)
            # #import pdb;pdb.set_trace()
            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            # the prune function relies on a feature of the class PriorityQueue. For this class the lowest value is always retrieved first and removed.
            # We assure that we read only the smallest, i.e. the most probable values, from the search beams that have not ended yet. After pruning the nodes attribute of the BeamSeachClass contains only nodes that represent an unfinished beam, and it contains beamsize - finished nodes
            # The nod stores the whole history of the beam. The log probs are added in every step. So when get_best is called the over all best nod is return and with it the whole beam

            for search in searches:
                search.prune()
            # Todo: The paper mentions that they always keep number of beamsize nodes active, no matter how many have terminated already
            # if not bool(args.nbest):
            #     for search in searches:
            #         search.prune()
            # else:
            #     for search in searches:
            #         # keep beamsize number of beams, no matter how many finished
            #         search.prune_n_best()
        # Segment into sentences
        if not bool(args.nbest):
            best_sents = torch.stack([search.get_best()[1].sequence[1:].cpu() for search in searches])
            decoded_batch = best_sents.numpy()
            #import pdb;pdb.set_trace()

            output_sentences = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

            # __QUESTION 6: What is the purpose of this for loop?
            # The loop serves to just append the real tokes of this specific output sentence for each sentence in outputsentences.
            # We masked all the sentences to have the same length, thus we need to make sure we don't output the mask token
            temp = list()
            for sent in output_sentences:
                first_eos = np.where(sent == tgt_dict.eos_idx)[0]
                if len(first_eos) > 0:
                    temp.append(sent[:first_eos[0]])
                else:
                    temp.append(sent)
            output_sentences = temp

            # Convert arrays of indices into strings of words
            output_sentences = [tgt_dict.string(sent) for sent in output_sentences]
            for ii, sent in enumerate(output_sentences):
                all_hyps[int(sample['id'].data[ii])] = sent
        else:
            # CHANGES
            for ii, search in enumerate(searches):
                all_hyps[int(sample['id'].data[ii])] = []
                best_sents = torch.stack([node[1].sequence[1:].cpu() for node in search.get_n_best(args.nbest)])
                decoded_batch = best_sents.numpy()
                # import pdb;pdb.set_trace()

                output_sentences = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

                # __QUESTION 6: What is the purpose of this for loop?
                temp = list()
                for sent in output_sentences:
                    first_eos = np.where(sent == tgt_dict.eos_idx)[0]
                    if len(first_eos) > 0:
                        temp.append(sent[:first_eos[0]])
                    else:
                        temp.append(sent)
                output_sentences = temp

                # Convert arrays of indices into strings of words
                output_sentences = [tgt_dict.string(sent) for sent in output_sentences]
                for sent in output_sentences:
                    all_hyps[int(sample['id'].data[ii])].append(sent)
    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            if not bool(args.nbest):
                for sent_id in range(len(all_hyps.keys())):
                    out_file.write(all_hyps[sent_id] + '\n')
            else:
                # CHANGES
                for sent_id in range(len(all_hyps.keys())):
                    out_file.write(str(sent_id) + '\n')
                    for c, sent in enumerate(all_hyps[sent_id]):
                        out_file.write(f"{sent_id}.{c+1}\t" + all_hyps[sent_id][c] + '\n')



if __name__ == '__main__':
    args = get_args()
    main(args)
