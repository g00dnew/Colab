# KASELL 2026 초록 초안 v2 (2026-10-12, 수치는 t12/results/SUMMARY.md "T1′ 통제 분석"·"단어 그룹 CV 조건의 주검정 확인" 절)

한 줄 노벨티: 본 연구의 새로움은 음소 표상의 위치 효과 자체가 아니라, 단어 내 위치가 서로 다른 음성 자질의 신경표상(해독 가능한 정보량)에 차등적으로 영향을 미치는지를 직접 검증했다는 데 있다.

확인할 것: KASELL 분량 제한 / "sequential position" 선행 문헌(없으면 해당 구절 삭제) / 공동 저자 순서.

---

**Laryngeal, but not place, information in human area 6v depends on position in the word**

English consonants keep their place of articulation wherever they fall in a word, but the laryngeal contrast does not: /p/ and /b/ are well separated word-initially and far less so in medial and final position. We asked whether this asymmetry is already present in the speech motor plan. If area 6v codes phonemes as position-independent units, laryngeal and place contrasts should be equally decodable in any position. If it codes what the vocal tract is about to do, laryngeal information should fall where the contrast is phonetically reduced, and place information should not.

We reanalysed the public intracortical recordings of participant T12 (Willett et al., 2023): 24 sessions and 9,680 attempted sentences from four microelectrode arrays in areas 6v and 44 of a person with ALS. Phoneme boundaries came from CTC forced alignment with the authors' pretrained decoder. In the isolated-phoneme task a 39-way classifier reached 64.5% in 6v (chance 2.6%) and 10.2% in area 44, so the remaining analyses used 6v. For each minimal pair we trained shrinkage-LDA classifiers on word-initial and on non-initial tokens, with counts matched across positions. Laryngeal pairs were /p–b/, /t–d/, /k–g/, plus /f–v/ and /θ–ð/ as a control for aspiration; place pairs were /p–t/, /t–k/, /b–d/, /d–g/.

Sentence data invite leakage, so the analysis plan was fixed before results were examined. PCA was fit inside each training fold. Tokens from one sentence never appeared on both sides of a split, and separate runs did the same for recording blocks, sessions, and words. In non-initial position the medial-to-final ratio was equalised between the two members of each pair. Because "to" alone supplies 64% of initial /t/ tokens, a further run capped the number of tokens per word.

Laryngeal discriminability fell from initial to non-initial position in every version of the analysis. With word-grouped cross-validation, laryngeal accuracy dropped from 60% to 52% while place accuracy rose from 58% to 65%. The feature × position interaction was D = 0.088 (95% range over 20 balanced resamplings [0.047, 0.139], positive in all 20), a segment-level permutation of position labels rejected the null at p = .008, and the direction held when any single pair was left out. With the per-word cap, D = 0.085 [0.036, 0.130]. The same interaction was present in activity 300–100 ms before each segment began (D = 0.097 [0.042, 0.153]). Fricative pairs, which carry no aspiration, declined as much as stops, so this is not a signature of voice onset time or [spread glottis]. Classifiers trained on initial tokens transferred poorly to non-initial tokens for both feature types: what separates the features is how much information survives the change of position, not whether the pattern itself is invariant. Mouthed speech, by contrast, reduced laryngeal and place information in proportion, with no selective laryngeal loss.

Area 6v therefore carries position-specific phonetic detail rather than a fixed phoneme code, and the laryngeal dimension is where that detail is lost. Decoders that treat /t/ as one unit regardless of position discard this structure. The study has one participant, arrays confined to ventral 6v, small effects, and /t–d/ contributing most of the laryngeal decline.
