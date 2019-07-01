from scipy.io import wavfile
import numpy
from os import path
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import peakutils as pu
from subprocess import Popen


def get_F_0(signal, rate, time_step=0.0, min_pitch=75, max_pitch=600,
            max_num_cands=15, silence_thres=.03, voicing_thres=.45,
            octave_cost=.01, octave_jump_cost=.35,
            voiced_unvoiced_cost=.14, accurate=False, pulse=False):
    """
    Computes median Fundamental Frequency ( :math:`F_0` ).
    The fundamental frequency ( :math:`F_0` ) of a signal is the lowest
    frequency, or the longest wavelength of a periodic waveform. In the context
    of this algorithm, :math:`F_0` is calculated by segmenting a signal into
    frames, then for each frame the most likely candidate is chosen from the
    lowest possible frequencies to be :math:`F_0`. From all of these values,
    the median value is returned. More specifically, the algorithm filters out
    frequencies higher than the Nyquist Frequency from the signal, then
    segments the signal into frames of at least 3 periods of the minimum
    pitch. For each frame, it then calculates the normalized autocorrelation
    ( :math:`r_a` ), or the correlation of the signal to a delayed copy of
    itself. :math:`r_a` is calculated according to Boersma's paper
    ( referenced below ), which is an improvement of previous methods.
    :math:`r_a` is estimated by dividing the autocorrelation of the windowed
    signal by the autocorrelation of the window. After :math:`r_a` is
    calculated the maxima values of :math:`r_a` are found. These points
    correspond to the lag domain, or points in the delayed signal, where the
    correlation value has peaked. The higher peaks indicate a stronger
    correlation. These points in the lag domain suggest places of wave
    repetition and are the candidates for :math:`F_0`. The best candidate for
    :math:`F_0` of each frame is picked by a cost function, a function that
    compares the cost of transitioning from the best :math:`F_0` of the
    previous frame to all possible :math:`F_0's` of the current frame. Once the
    path of :math:`F_0's` of least cost has been determined, the median
    :math:`F_0` of all voiced frames is returned.
    This algorithm is adapted from:
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    and from:
    https://github.com/praat/praat/blob/master/fon/Sound_to_Pitch.cpp

    .. note::
        It has been shown that depressed and suicidal men speak with a reduced
        fundamental frequency range, ( described in:
        http://ameriquests.org/index.php/vurj/article/download/2783/1181 ) and
        patients responding well to depression treatment show an increase in
        their fundamental frequency variability ( described in :
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3022333/ ). Because
        acoustical properties of speech are the earliest and most consistent
        indicators of mood disorders, early detection of fundamental frequency
        changes could significantly improve recovery time for disorders with
        psychomotor symptoms.

    Args:
        signal ( numpy.ndarray ): This is the signal :math:`F_0` will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        time_step ( float ): ( optional, default value: 0.0 ) The measurement, in seconds, of time passing between each frame. The smaller the time_step, the more overlap that will occur. If 0 is supplied the degree of oversampling will be equal to four.
        min_pitch ( float ): ( optional, default value: 75 ) This is the minimum value to be returned as pitch, which cannot be less than or equal to zero.
        max_pitch ( float ): ( optional, default value: 600 ) This is the maximum value to be returned as pitch, which cannot be greater than the Nyquist Frequency.
        max_num_cands ( int ): ( optional, default value: 15 ) This is the maximum number of candidates to be considered for each frame, the unvoiced candidate ( i.e. :math:`F_0` equal to zero ) is always considered.
        silence_thres ( float ): ( optional, default value: 0.03 ) Frames that do not contain amplitudes above this threshold ( relative to the global maximum amplitude ), are probably silent.
        voicing_thres ( float ): ( optional, default value: 0.45 ) This is the strength of the unvoiced candidate, relative to the maximum possible :math:`r_a`. To increase the number of unvoiced decisions, increase this value.
        octave_cost ( float ): ( optional, default value: 0.01 per octave ) This is the degree of favouring of high-frequency candidates, relative to the maximum possible :math:`r_a`. This is necessary because in the case of a perfectly periodic signal, all undertones of :math:`F_0` are equally strong candidates as :math:`F_0` itself. To more strongly favour recruitment of high-frequency candidates, increase this value.
        octave_jump_cost ( float ): ( optional, default value: 0.35 ) This is degree of disfavouring of pitch changes, relative to the maximum possible :math:`r_a`. To decrease the number of large frequency jumps, increase this value.
        voiced_unvoiced_cost ( float ): ( optional, default value: 0.14 ) This is the degree of disfavouring of voiced/unvoiced transitions, relative to the maximum possible :math:`r_a`. To decrease the number of voiced/unvoiced transitions, increase this value.
        accurate ( bool ): ( optional, default value: False ) If False, the window is a Hanning window with a length of :math:`\\frac{ 3.0} {min\_pitch}`. If True, the window is a Gaussian window with a length of :math:`\\frac{6.0}{min\_pitch}`, i.e. twice the length.
        pulse ( bool ): ( optional, default value: False ) If False, the function returns a list containing only the median :math:`F_0`. If True, the function returns a list with all values necessary to calculate pulses. This list contains the median :math:`F_0`, the frequencies for each frame in a list, a list of tuples containing the beginning time of the frame, and the ending time of the frame, and the signal filtered by the Nyquist Frequency. The indicies in the second and third list correspond to each other.

    Returns:
        list: Index 0 contains the median :math:`F_0` in hz. If pulse is set
        equal to True, indicies 1, 2, and 3 will contain: a list of all voiced
        periods in order, a list of tuples of the beginning and ending time
        of a voiced interval, with each index in the list corresponding to the
        previous list, and a numpy.ndarray of the signal filtered by the
        Nyquist Frequency. If pulse is set equal to False, or left to the
        default value, then the list will only contain the median :math:`F_0`.

    Raises:
        ValueError: min_pitch has to be greater than zero.
        ValueError: octave_cost isn't in [ 0, 1 ].
        ValueError: silence_thres isn't in [ 0, 1 ].
        ValueError: voicing_thres isn't in [ 0, 1 ].
        ValueError: max_pitch can't be larger than Nyquist Frequency.



    .. figure::  figures/F_0_synthesized_sig.png
       :align:   center
    """
    if min_pitch <= 0:
        raise ValueError("min_pitch has to be greater than zero.")

    if max_num_cands < max_pitch / min_pitch:
        max_num_cands = int(max_pitch / min_pitch)

    initial_len = len(signal)
    total_time = initial_len / float(rate)
    tot_time_arr = np.linspace(0, total_time, initial_len)
    max_place_poss = 1.0 / min_pitch
    min_place_poss = 1.0 / max_pitch
    # to silence formants
    min_place_poss2 = 0.5 / max_pitch

    if accurate:
        pds_per_window = 6.0
    else:
        pds_per_window = 3.0

    # degree of oversampling is 4
    if time_step <= 0: time_step = (pds_per_window / 4.0) / min_pitch

    w_len = pds_per_window / min_pitch
    # correcting for time_step
    octave_jump_cost *= .01 / time_step
    voiced_unvoiced_cost *= .01 / time_step

    Nyquist_Frequency = rate / 2.0
    upper_bound = .95 * Nyquist_Frequency
    zeros_pad = 2 ** (int(np.log2(initial_len)) + 1) - initial_len
    signal = np.hstack((signal, np.zeros(zeros_pad)))
    fft_signal = np.fft.fft(signal)
    fft_signal[int(upper_bound): -int(upper_bound)] = 0
    sig = np.fft.ifft(fft_signal)
    sig = sig[:initial_len].real

    # checking to make sure values are valid
    if Nyquist_Frequency < max_pitch:
        raise ValueError("max_pitch can't be larger than Nyquist Frequency.")
    if octave_cost < 0 or octave_cost > 1:
        raise ValueError("octave_cost isn't in [ 0, 1 ]")
    if voicing_thres < 0 or voicing_thres > 1:
        raise ValueError("voicing_thres isn't in [ 0, 1 ]")
    if silence_thres < 0 or silence_thres > 1:
        raise ValueError("silence_thres isn't in [ 0, 1 ]")

    # finding number of samples per frame and time_step
    frame_len = int(w_len * rate + .5)
    time_len = int(time_step * rate + .5)

    # initializing list of candidates for F_0, and their strengths
    best_cands, strengths, time_vals = [], [], []

    # finding the global peak the way Praat does
    global_peak = max(abs(sig - sig.mean()))
    print(type(global_peak), '\n')
    e = np.e
    inf = np.inf
    log = np.log2
    start_i = 0
    while start_i < len(sig) - frame_len:
        end_i = start_i + frame_len
        segment = sig[start_i: end_i]

        if accurate:
            t = np.linspace(0, w_len, len(segment))
            numerator = e ** (-12.0 * (t / w_len - .5) ** 2.0) - e ** -12.0
            denominator = 1.0 - e ** -12.0
            window = numerator / denominator
            interpolation_depth = 0.25
        else:
            window = np.hanning(len(segment))
            interpolation_depth = 0.50

        # shave off ends of time intervals to account for overlapping
        start_time = tot_time_arr[start_i + int(time_len / 4.0)]
        stop_time = tot_time_arr[end_i - int(time_len / 4.0)]
        time_vals.append((start_time, stop_time))

        start_i += time_len

        long_pd_i = int(rate / min_pitch)
        half_pd_i = int(long_pd_i / 2.0 + 1)

        long_pd_cushion = segment[half_pd_i: - half_pd_i]
        # finding local peak and local mean the way Praat does
        # local mean is found by looking a longest period to either side of the
        # center of the frame, and using only the values within this interval to
        # calculate the local mean, and similarly local peak is found by looking
        # a half of the longest period to either side of the center of the
        # frame, ( after the frame has windowed ) and choosing the absolute
        # maximum in this interval
        local_mean = long_pd_cushion.mean()
        segment = segment - local_mean
        segment *= window
        half_pd_cushion = segment[long_pd_i: -long_pd_i]
        local_peak = max(abs(half_pd_cushion))
        if local_peak == 0:
            # shortcut -> complete silence and only candidate is silent candidate
            best_cands.append([inf])
            strengths.append([voicing_thres + 2])
        else:
            # calculating autocorrelation, based off steps 3.2-3.10
            intensity = local_peak / float(global_peak)

            N = len(segment)
            nFFT = 2 ** int(log((1.0 + interpolation_depth) * N) + 1)
            window = np.hstack((window, np.zeros(nFFT - N)))
            segment = np.hstack((segment, np.zeros(nFFT - N)))
            x_fft = np.fft.fft(segment)
            r_a = np.real(np.fft.fft(x_fft * np.conjugate(x_fft)))
            r_a = r_a[: int(N / pds_per_window)]

            x_fft = np.fft.fft(window)
            r_w = np.real(np.fft.fft(x_fft * np.conjugate(x_fft)))
            r_w = r_w[: int(N / pds_per_window)]
            r_x = r_a / r_w
            r_x /= r_x[0]

            # creating an array of the points in time corresponding to sampled
            # autocorrelation of the signal ( r_x )
            time_array = np.linspace(0, w_len / pds_per_window, len(r_x))
            peaks = pu.indexes(r_x, thres=0)
            max_values, max_places = r_x[peaks], time_array[peaks]

            # only consider places that are voiced over a certain threshold
            max_places = max_places[max_values > 0.5 * voicing_thres]
            max_values = max_values[max_values > 0.5 * voicing_thres]

            for i in range(len(max_values)):
                # reflecting values > 1 through 1.
                if max_values[i] > 1.0:
                    max_values[i] = 1.0 / max_values[i]

            # calculating the relative strength value
            rel_val = [val - octave_cost * log(place * min_pitch) for
                       val, place in zip(max_values, max_places)]

            if len(max_values) > 0.0:
                # finding the max_num_cands-1 maximizers, and maximums, then
                # calculating their strengths ( eq. 23 and 24 ) and accounting for
                # silent candidate
                max_places = [max_places[i] for i in np.argsort(rel_val)[
                                                     -max_num_cands + 1:]]
                max_values = [max_values[i] for i in np.argsort(rel_val)[
                                                     -max_num_cands + 1:]]
                max_places = np.array(max_places)
                max_values = np.array(max_values)

                rel_val = list(np.sort(rel_val)[-max_num_cands + 1:])
                # adding the silent candidate's strength to strengths
                rel_val.append(voicing_thres + max(0, 2 - (intensity /
                                                           (silence_thres / (1 + voicing_thres)))))

                # inf is our silent candidate
                max_places = np.hstack((max_places, inf))

                best_cands.append(list(max_places))
                strengths.append(rel_val)
            else:
                # if there are no available maximums, only account for silent
                # candidate
                best_cands.append([inf])
                strengths.append([voicing_thres + max(0, 2 - intensity /
                                                      (silence_thres / (1 + voicing_thres)))])

    # Calculates smallest costing path through list of candidates ( forwards ),
    # and returns path.
    best_total_cost, best_total_path = -inf, []
    # for each initial candidate find the path of least cost, then of those
    # paths, choose the one with the least cost.
    for cand in range(len(best_cands[0])):
        start_val = best_cands[0][cand]
        total_path = [start_val]
        level = 1
        prev_delta = strengths[0][cand]
        maximum = -inf
        while level < len(best_cands):
            prev_val = total_path[-1]
            best_val = inf
            for j in range(len(best_cands[level])):
                cur_val = best_cands[level][j]
                cur_delta = strengths[level][j]
                cost = 0
                cur_unvoiced = cur_val == inf or cur_val < min_place_poss2
                prev_unvoiced = prev_val == inf or prev_val < min_place_poss2

                if cur_unvoiced:
                    # both voiceless
                    if prev_unvoiced:
                        cost = 0
                        # voiced-to-unvoiced transition
                    else:
                        cost = voiced_unvoiced_cost
                else:
                    # unvoiced-to-voiced transition
                    if prev_unvoiced:
                        cost = voiced_unvoiced_cost
                    # both are voiced
                    else:
                        cost = octave_jump_cost * abs(log(cur_val /
                                                          prev_val))

                # The cost for any given candidate is given by the transition
                # cost, minus the strength of the given candidate
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val

            prev_delta = maximum
            total_path.append(best_val)
            level += 1

        if maximum > best_total_cost:
            best_total_cost, best_total_path = maximum, total_path

    f_0_forth = np.array(best_total_path)

    # Calculates smallest costing path through list of candidates ( backwards ),
    # and returns path. Going through the path backwards introduces frequencies
    # previously marked as unvoiced, or increases undertones, to decrease
    # frequency jumps

    best_total_cost, best_total_path2 = -inf, []

    # Starting at the end, for each initial candidate find the path of least
    # cost, then of those paths, choose the one with the least cost.
    for cand in range(len(best_cands[-1])):
        start_val = best_cands[-1][cand]
        total_path = [start_val]
        level = len(best_cands) - 2
        prev_delta = strengths[-1][cand]
        maximum = -inf
        while level > -1:
            prev_val = total_path[-1]
            best_val = inf
            for j in range(len(best_cands[level])):
                cur_val = best_cands[level][j]
                cur_delta = strengths[level][j]
                cost = 0
                cur_unvoiced = cur_val == inf or cur_val < min_place_poss2
                prev_unvoiced = prev_val == inf or prev_val < min_place_poss2

                if cur_unvoiced:
                    # both voiceless
                    if prev_unvoiced:
                        cost = 0
                        # voiced-to-unvoiced transition
                    else:
                        cost = voiced_unvoiced_cost
                else:
                    # unvoiced-to-voiced transition
                    if prev_unvoiced:
                        cost = voiced_unvoiced_cost
                    # both are voiced
                    else:
                        cost = octave_jump_cost * abs(log(cur_val /
                                                          prev_val))

                # The cost for any given candidate is given by the transition
                # cost, minus the strength of the given candidate
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val

            prev_delta = maximum
            total_path.append(best_val)
            level -= 1

        if maximum > best_total_cost:
            best_total_cost, best_total_path2 = maximum, total_path

    f_0_back = np.array(best_total_path2)
    # reversing f_0_backward so the initial value corresponds to first frequency
    f_0_back = f_0_back[-1:: -1]

    # choose the maximum frequency from each path for the total path
    f_0 = np.array([min(i, j) for i, j in zip(f_0_forth, f_0_back)])

    if pulse:
        # removing all unvoiced time intervals from list
        removed = 0
        for i in range(len(f_0)):
            if f_0[i] > max_place_poss or f_0[i] < min_place_poss:
                time_vals.remove(time_vals[i - removed])
                removed += 1

    for i in range(len(f_0)):
        # if f_0 is voiceless assign occurance of peak to inf -> when divided
        # by one this will give us a frequency of 0, corresponding to a unvoiced
        # frame
        if f_0[i] > max_place_poss or f_0[i] < min_place_poss:
            f_0[i] = inf

    f_0 = f_0[f_0 < inf]
    if pulse:
        return [np.median(1.0 / f_0), list(f_0), time_vals, signal]
    if len(f_0) == 0:
        return [0]
    else:
        return [np.median(1.0 / f_0)]

while 1:
    p = Popen("downlaodAws.bat", cwd=r"C:\dev\TCC_2_Pedro\TCC_2_Pedro\Upload_AWS\Testaws")
    stdout, stderr = p.communicate()

    # files                                                                         
    src = "pedro_fox.3gp"
    dst = "test.wav"
    #Limpando o arquivo TXT
    arquivo = open('freq_o.txt', 'w')
    arquivo.close()

    # convert wav to mp3                                                            
    sound = AudioSegment.from_file(src)
    sound.export(dst, format="wav")
    #faz a leitura do arquivo wav
    signal,samplerate = sf.read("test.wav")
    #Função do calcululo da Frequencia fundamental
    x = get_F_0(signal,samplerate)
    #transforma o arquivo em String
    frequ_o = str(x)
    #Escreve no arquivo Txt
    arquivo = open('freq_o.txt','w')
    arquivo.write(frequ_o)
    arquivo.close()

    p = Popen("clearAws.bat", cwd=r"C:\dev\TCC_2_Pedro\TCC_2_Pedro\Upload_AWS\Testaws")
    stdout, stderr = p.communicate()

