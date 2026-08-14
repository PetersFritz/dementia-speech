/**
 * Bound Apps Script for the study's consent Google Form.
 *
 * On each submission: generates an opaque token, records it against the
 * response row in the linked Sheet (so consent records stay linkable to a
 * token without the token needing to leave Google), and emails the
 * respondent their personal study link together with a copy of the
 * participant information sheet and a record of the consent answers they
 * just submitted.
 *
 * Must run as an INSTALLABLE trigger, not a simple trigger — simple triggers
 * (a bare function named onFormSubmit) cannot call MailApp. See
 * study/README.md for how to add the installable trigger.
 */

// --- Configuration — fill these in before going live ---
const STUDY_BASE_URL = 'https://REPLACE_WITH_PUBLIC_HOST/'; // study_app.py public URL, see docs/study_hosting.md
const STUDY_NAME = 'Dementia Speech Dashboard — Clinical Decision Study';
const SENDER_NAME = 'REPLACE_WITH_YOUR_NAME';
// Drive file ID of the finalised participant information sheet PDF (upload
// it to Drive, open it, and copy the ID out of the file's URL). Must be
// readable by whichever Google account runs this script.
const INFO_SHEET_PDF_ID = 'REPLACE_WITH_DRIVE_FILE_ID';

function onFormSubmitHandler(e) {
  const response = e.response;
  const email = response.getRespondentEmail();
  if (!email) {
    throw new Error(
      'No respondent email captured — enable "Collect email addresses" in the Form\'s ' +
      'Settings > Responses, set to "Responder input" (not "Verified", which would ' +
      'require the participant to sign in with a Google account).'
    );
  }

  const token = Utilities.getUuid();
  recordToken_(response, token);
  sendStudyLinkEmail_(email, token, response);
}

/**
 * Writes the token into a "token" column on the response's row in the
 * linked Sheet (creating the column on first use), matched by timestamp.
 */
function recordToken_(response, token) {
  const form = FormApp.getActiveForm();
  const sheet = SpreadsheetApp.openById(form.getDestinationId()).getSheets()[0];
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];

  let tokenCol = headers.indexOf('token') + 1;
  if (tokenCol === 0) {
    tokenCol = sheet.getLastColumn() + 1;
    sheet.getRange(1, tokenCol).setValue('token');
  }

  const responseTimestamp = response.getTimestamp().getTime();
  const timestamps = sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getValues();
  for (let i = 0; i < timestamps.length; i++) {
    if (timestamps[i][0] && timestamps[i][0].getTime() === responseTimestamp) {
      sheet.getRange(i + 2, tokenCol).setValue(token);
      break;
    }
  }
}

function sendStudyLinkEmail_(email, token, response) {
  const link = STUDY_BASE_URL + '?token=' + encodeURIComponent(token);
  const subject = STUDY_NAME + ' — your personal study link';
  const body = [
    'Thank you for consenting to take part in the study.',
    '',
    'Your personal link to the study (please do not share it — it identifies your responses):',
    link,
    '',
    'The study presents 40 cases. After the final case you will be asked to complete a short',
    'follow-up questionnaire. You can close your browser at any point and reopen this same',
    'link later to continue where you left off.',
    '',
    'Attached is a copy of the participant information sheet for your records.',
    '',
    'For your records, here is a copy of the consent responses you submitted just now:',
    formatConsentRecord_(response),
    '',
    'Thank you,',
    SENDER_NAME,
  ].join('\n');

  const infoSheetPdf = DriveApp.getFileById(INFO_SHEET_PDF_ID).getAs(MimeType.PDF);
  MailApp.sendEmail({
    to: email,
    subject: subject,
    body: body,
    attachments: [infoSheetPdf],
  });
}

/**
 * Renders the respondent's answers to every form item (information-sheet
 * acknowledgement and consent statements alike) as a plain-text list, so the
 * confirmation email doubles as their personal record of what they agreed
 * to.
 */
function formatConsentRecord_(response) {
  return response.getItemResponses().map(function (itemResponse) {
    const question = itemResponse.getItem().getTitle();
    const answer = itemResponse.getResponse();
    const answerText = Array.isArray(answer) ? answer.join('; ') : answer;
    return '- ' + question + ': ' + answerText;
  }).join('\n');
}
